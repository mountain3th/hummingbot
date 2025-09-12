import os
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Set

import pandas as pd
from pydantic import Field, field_validator

from hummingbot.client.ui.interface_utils import format_df_for_printout
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.connector.utils import split_hb_trading_pair
from hummingbot.core.clock import Clock
from hummingbot.core.data_type.common import OrderType, PositionAction, PositionMode, PriceType, TradeType
from hummingbot.core.data_type.funding_info import FundingInfo
from hummingbot.core.data_type.order_candidate import PerpetualOrderCandidate
from hummingbot.core.event.events import FundingPaymentCompletedEvent
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy.strategy_v2_base import StrategyV2Base, StrategyV2ConfigBase
from hummingbot.strategy_v2.executors.position_executor.data_types import PositionExecutorConfig, TripleBarrierConfig
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, StopExecutorAction


class FundingRateArbitrageConfig(StrategyV2ConfigBase):
    script_file_name: str = os.path.basename(__file__)
    candles_config: List[CandlesConfig] = []
    controllers_config: List[str] = []
    markets: Dict[str, Set[str]] = {}
    leverage: int = Field(
        default=20, gt=0,
        json_schema_extra={"prompt": lambda mi: "Enter the leverage (e.g. 20): ", "prompt_on_new": True},
    )
    min_funding_rate_profitability: Decimal = Field(
        default=0.001,
        json_schema_extra={
            "prompt": lambda mi: "Enter the min funding rate profitability to enter in a position (e.g. 0.001): ",
            "prompt_on_new": True}
    )
    connectors: Set[str] = Field(
        default="hyperliquid_perpetual,binance_perpetual",
        json_schema_extra={
            "prompt": lambda mi: "Enter the connectors separated by commas (e.g. hyperliquid_perpetual,binance_perpetual): ",
            "prompt_on_new": True}
    )
    tokens: Set[str] = Field(
        default="WIF,FET",
        json_schema_extra={"prompt": lambda mi: "Enter the tokens separated by commas (e.g. WIF,FET): ", "prompt_on_new": True},
    )
    position_size_quote: Decimal = Field(
        default=100,
        json_schema_extra={
            "prompt": lambda mi: "Enter the position size in quote asset (e.g. order amount 100 will open 100 long on hyperliquid and 100 short on binance): ",
            "prompt_on_new": True
        }
    )
    profitability_to_take_profit: Decimal = Field(
        default=0.01,
        json_schema_extra={
            "prompt": lambda mi: "Enter the profitability to take profit (including PNL of positions and fundings received): ",
            "prompt_on_new": True}
    )
    profitability_to_early_stop: Decimal = Field(
        default=0.001,
        json_schema_extra={
            "prompt": lambda mi: "Enter the profitability to early stop the position: ",
            "prompt_on_new": True}
    )
    funding_rate_diff_stop_loss: Decimal = Field(
        default=-0.001,
        json_schema_extra={
            "prompt": lambda mi: "Enter the funding rate difference to stop the position (e.g. -0.001): ",
            "prompt_on_new": True}
    )
    trade_profitability_condition_to_enter: bool = Field(
        default=False,
        json_schema_extra={
            "prompt": lambda mi: "Do you want to check the trade profitability condition to enter? (True/False): ",
            "prompt_on_new": True}
    )
    open_new_position_delay: int = Field(
        default = 60 * 30,
        json_schema_extra={
            "prompt": lambda mi: "Enter the delay in seconds to open a new position after the last one was closed (e.g. 60): ",
            "prompt_on_new": True}
    )

    @field_validator("connectors", "tokens", mode="before")
    @classmethod
    def validate_sets(cls, v):
        if isinstance(v, str):
            return set(v.split(","))
        return v


class FundingRateArbitrage(StrategyV2Base):
    quote_markets_map = {
        "hyperliquid_perpetual": "USD",
        "binance_perpetual": "USDT"
    }
    funding_payment_interval_map = {
        "binance_perpetual": 60 * 60 * 8,
        "hyperliquid_perpetual": 60 * 60 * 1
    }
    funding_profitability_interval = 60 * 60 * 24

    @classmethod
    def get_trading_pair_for_connector(cls, token, connector):
        return f"{token}-{cls.quote_markets_map.get(connector, 'USDT')}"

    @classmethod
    def init_markets(cls, config: FundingRateArbitrageConfig):
        markets = {}
        for connector in config.connectors:
            trading_pairs = {cls.get_trading_pair_for_connector(token, connector) for token in config.tokens}
            markets[connector] = trading_pairs
        cls.markets = markets

    def __init__(self, connectors: Dict[str, ConnectorBase], config: FundingRateArbitrageConfig):
        super().__init__(connectors, config)
        self.config = config
        self.active_funding_arbitrages = {}
        self.stopped_funding_arbitrages = {token: [] for token in self.config.tokens}

    def start(self, clock: Clock, timestamp: float) -> None:
        """
        Start the strategy.
        :param clock: Clock to use.
        :param timestamp: Current time.
        """
        self._last_timestamp = timestamp
        self.closed_funding_time = {}
        self.apply_initial_setting()

    def apply_initial_setting(self):
        for connector_name, connector in self.connectors.items():
            if self.is_perpetual(connector_name):
                position_mode = PositionMode.ONEWAY if connector_name == "hyperliquid_perpetual" else PositionMode.HEDGE
                connector.set_position_mode(position_mode)
                for trading_pair in self.market_data_provider.get_trading_pairs(connector_name):
                    connector.set_leverage(trading_pair, self.config.leverage)

    def get_funding_info_by_token(self, token):
        """
        This method provides the funding rates across all the connectors
        """
        funding_rates = {}
        for connector_name, connector in self.connectors.items():
            trading_pair = self.get_trading_pair_for_connector(token, connector_name)
            funding_rates[connector_name] = connector.get_funding_info(trading_pair)
        return funding_rates

    def get_current_profitability_after_fees(self, token: str, connector_1: str, connector_2: str, side: TradeType):
        """
        This methods compares the profitability of buying at market in the two exchanges. If the side is TradeType.BUY
        means that the operation is long on connector 1 and short on connector 2.
        """
        trading_pair_1 = self.get_trading_pair_for_connector(token, connector_1)
        trading_pair_2 = self.get_trading_pair_for_connector(token, connector_2)

        connector_1_price = Decimal(self.market_data_provider.get_price_for_quote_volume(
            connector_name=connector_1,
            trading_pair=trading_pair_1,
            quote_volume=self.config.position_size_quote,
            is_buy=side == TradeType.BUY,
        ).result_price)
        connector_2_price = Decimal(self.market_data_provider.get_price_for_quote_volume(
            connector_name=connector_2,
            trading_pair=trading_pair_2,
            quote_volume=self.config.position_size_quote,
            is_buy=side != TradeType.BUY,
        ).result_price)
        estimated_fees_connector_1 = self.connectors[connector_1].get_fee(
            base_currency=trading_pair_1.split("-")[0],
            quote_currency=trading_pair_1.split("-")[1],
            order_type=OrderType.MARKET,
            order_side=TradeType.BUY,
            amount=self.config.position_size_quote / connector_1_price,
            price=connector_1_price,
            is_maker=False,
            position_action=PositionAction.OPEN
        ).percent
        estimated_fees_connector_2 = self.connectors[connector_2].get_fee(
            base_currency=trading_pair_2.split("-")[0],
            quote_currency=trading_pair_2.split("-")[1],
            order_type=OrderType.MARKET,
            order_side=TradeType.BUY,
            amount=self.config.position_size_quote / connector_2_price,
            price=connector_2_price,
            is_maker=False,
            position_action=PositionAction.OPEN
        ).percent

        if side == TradeType.BUY:
            estimated_trade_pnl_pct = (connector_2_price - connector_1_price) / connector_1_price
        else:
            estimated_trade_pnl_pct = (connector_1_price - connector_2_price) / connector_2_price

        return estimated_trade_pnl_pct - estimated_fees_connector_1 - estimated_fees_connector_2

    def get_most_profitable_combination(self, funding_info_report: Dict):
        best_combination = None
        highest_profitability = 0
        for connector_1 in funding_info_report:
            for connector_2 in funding_info_report:
                if connector_1 != connector_2:
                    rate_connector_1 = self.get_normalized_funding_rate_in_seconds(funding_info_report, connector_1)
                    rate_connector_2 = self.get_normalized_funding_rate_in_seconds(funding_info_report, connector_2)
                    funding_rate_diff = abs(rate_connector_1 - rate_connector_2) * self.funding_profitability_interval
                    if funding_rate_diff > highest_profitability:
                        trade_side = TradeType.BUY if rate_connector_1 < rate_connector_2 else TradeType.SELL
                        highest_profitability = funding_rate_diff
                        best_combination = (connector_1, connector_2, trade_side, funding_rate_diff)
        return best_combination

    def get_normalized_funding_rate_in_seconds(self, funding_info_report: Dict[str, FundingInfo], connector_name: str):
        if funding_info_report[connector_name].funding_interval is None:
            return funding_info_report[connector_name].rate / self.funding_payment_interval_map.get(connector_name, 60 * 60 * 8)
        return funding_info_report[connector_name].rate / Decimal(funding_info_report[connector_name].funding_interval)

    def validate_sufficient_balance(self, executor_config1: PositionExecutorConfig, executor_config2: PositionExecutorConfig) -> bool:
        price1 = self.market_data_provider.get_price_for_quote_volume(
            connector_name=executor_config1.connector_name,
            trading_pair=executor_config1.trading_pair,
            quote_volume=self.config.position_size_quote,
            is_buy=executor_config1.side == TradeType.BUY
        )
        price2 = self.market_data_provider.get_price_for_quote_volume(
            connector_name=executor_config2.connector_name,
            trading_pair=executor_config2.trading_pair,
            quote_volume=self.config.position_size_quote,
            is_buy=executor_config2.side == TradeType.BUY
        )
        order_candidate1 = PerpetualOrderCandidate(
            trading_pair=executor_config1.trading_pair,
            is_maker=False,
            order_type=OrderType.MARKET,
            order_side=executor_config1.side,
            amount=executor_config1.amount,
            price=Decimal(price1.result_price),
            leverage=executor_config1.leverage,
        )
        order_candidate2 = PerpetualOrderCandidate(
            trading_pair=executor_config2.trading_pair,
            is_maker=False,
            order_type=OrderType.MARKET,
            order_side=executor_config2.side,
            amount=executor_config2.amount,
            price=Decimal(price2.result_price),
            leverage=executor_config2.leverage,
        )
        order_adjusted_candidate1 = self.connectors[executor_config1.connector_name].budget_checker.adjust_candidate(
            order_candidate1, all_or_none=True
        )
        order_adjusted_candidate2 = self.connectors[executor_config2.connector_name].budget_checker.adjust_candidate(
            order_candidate2, all_or_none=True
        )
        if order_adjusted_candidate1.is_zero_order or order_adjusted_candidate2.is_zero_order:
            return False
        return True

    def validate_minimum_order_size(self, executor_config: PositionExecutorConfig) -> bool:
        trading_rule = self.connectors[executor_config.connector_name].trading_rules[executor_config.trading_pair]
        return executor_config.amount >= trading_rule.min_order_size

    def get_close_fee(self, connector_name: str, trading_pair: str, order_type: OrderType, order_side: TradeType, amount: Decimal) -> Decimal:
        """
        This method returns the fee for the given connector, trading pair, order type, order side, amount and price.
        """
        fee = self.connectors[connector_name].get_fee(
            base_currency=trading_pair.split("-")[0],
            quote_currency=trading_pair.split("-")[1],
            order_type=order_type,
            order_side=order_side,
            amount=amount,
            is_maker=False,
            position_action=PositionAction.CLOSE
        )
        return fee.percent

    def create_actions_proposal(self) -> List[CreateExecutorAction]:
        """
        In this method we are going to evaluate if a new set of positions has to be created for each of the tokens that
        don't have an active arbitrage.
        More filters can be applied to limit the creation of the positions, since the current logic is only checking for
        positive pnl between funding rate. Is logged and computed the trading profitability at the time for entering
        at market to open the possibilities for other people to create variations like sending limit position executors
        and if one gets filled buy market the other one to improve the entry prices.
        """
        create_actions = []
        for token in self.config.tokens:
            if token not in self.active_funding_arbitrages and self.current_timestamp - self.closed_funding_time.get(token, 0) > self.config.open_new_position_delay:
                funding_info_report = self.get_funding_info_by_token(token)
                best_combination = self.get_most_profitable_combination(funding_info_report)
                if not best_combination:
                    continue
                connector_1, connector_2, trade_side, expected_profitability = best_combination
                if expected_profitability >= self.config.min_funding_rate_profitability:
                    current_profitability = self.get_current_profitability_after_fees(
                        token, connector_1, connector_2, trade_side
                    )
                    if self.config.trade_profitability_condition_to_enter:
                        if current_profitability < 0:
                            continue
                    self.logger().info(f"Best Combination-{token}: {connector_1} | {connector_2} | {trade_side}"
                                       f" Funding rate profitability: {expected_profitability}"
                                       f" Trading profitability after fees: {current_profitability}"
                                       f" Starting executors...")
                    position_executor_config_1, position_executor_config_2 = self.get_position_executors_config(token, connector_1, connector_2, trade_side)
                    if not position_executor_config_1 or not position_executor_config_2:
                        continue
                    if not self.validate_sufficient_balance(position_executor_config_1, position_executor_config_2):
                        self.logger().error(f"Not enough budget to open position for {token} on {connector_1} and {connector_2}.")
                        continue
                    if not self.validate_minimum_order_size(position_executor_config_1) or not self.validate_minimum_order_size(position_executor_config_2):
                        self.logger().error(f"Minimum order size not met for {token} on {connector_1} and {connector_2}.")
                        continue
                    self.active_funding_arbitrages[token] = {
                        "connector_1": connector_1,
                        "connector_2": connector_2,
                        "executors_ids": [position_executor_config_1.id, position_executor_config_2.id],
                        "funding_payments": [],
                        "side": trade_side,
                    }
                    return [CreateExecutorAction(executor_config=position_executor_config_1),
                            CreateExecutorAction(executor_config=position_executor_config_2)]
        return create_actions

    def stop_actions_proposal(self) -> List[StopExecutorAction]:
        """
        Once the funding rate arbitrage is created we are going to control the funding payments pnl and the current
        pnl of each of the executors at the cost of closing the open position at market.
        If that PNL is greater than the profitability_to_take_profit
        """
        stop_executor_actions = []
        stop_tokens = []
        for token, funding_arbitrage_info in self.active_funding_arbitrages.items():
            executors = self.filter_executors(
                executors=self.get_all_executors(),
                filter_func=lambda x: x.id in funding_arbitrage_info["executors_ids"]
            )
            funding_payments_pnl = sum(funding_payment.amount for funding_payment in funding_arbitrage_info["funding_payments"])
            executors_pnl = sum(executor.net_pnl_quote for executor in executors)
            close_position_net_pnl_pct = sum(
                executor.net_pnl_pct - self.get_close_fee(
                    connector_name=executor.connector_name,
                    trading_pair=executor.trading_pair,
                    order_type=OrderType.MARKET,
                    order_side=executor.side,
                    amount=Decimal('0')
                ) for executor in executors)

            reversion_profit_condition = close_position_net_pnl_pct > self.config.profitability_to_early_stop
            take_profit_condition = executors_pnl + funding_payments_pnl > self.config.profitability_to_take_profit * self.config.position_size_quote
            connector1_last_rate = next((payment.funding_rate for payment in reversed(funding_arbitrage_info["funding_payments"]) if split_hb_trading_pair(payment.trading_pair)[0] == token and payment.market == funding_arbitrage_info["connector_1"]), None)
            connector2_last_rate = next((payment.funding_rate for payment in reversed(funding_arbitrage_info["funding_payments"]) if split_hb_trading_pair(payment.trading_pair)[0] == token and payment.market == funding_arbitrage_info["connector_2"]), None)
            if connector1_last_rate and connector2_last_rate:
                funding_info_report = self.get_funding_info_by_token(token)
                funding_info_report[funding_arbitrage_info["connector_1"]].rate = connector1_last_rate
                funding_info_report[funding_arbitrage_info["connector_2"]].rate = connector2_last_rate
                if funding_arbitrage_info["side"] == TradeType.BUY:
                    funding_rate_diff = self.get_normalized_funding_rate_in_seconds(funding_info_report, funding_arbitrage_info["connector_2"]) - self.get_normalized_funding_rate_in_seconds(funding_info_report, funding_arbitrage_info["connector_1"])
                else:
                    funding_rate_diff = self.get_normalized_funding_rate_in_seconds(funding_info_report, funding_arbitrage_info["connector_1"]) - self.get_normalized_funding_rate_in_seconds(funding_info_report, funding_arbitrage_info["connector_2"])
                current_funding_condition = funding_rate_diff * self.funding_profitability_interval < self.config.funding_rate_diff_stop_loss
            else:
                current_funding_condition = False

            if take_profit_condition:
                self.logger().info("Take profit profitability reached, stopping executors")
                self.stopped_funding_arbitrages[token].append(funding_arbitrage_info)
                stop_executor_actions.extend([StopExecutorAction(executor_id=executor.id) for executor in executors])
                stop_tokens.append(token)
            elif reversion_profit_condition:
                self.logger().info("Reversion profit condition reached, stopping executors")
                self.stopped_funding_arbitrages[token].append(funding_arbitrage_info)
                stop_executor_actions.extend([StopExecutorAction(executor_id=executor.id) for executor in executors])
                stop_tokens.append(token)
            elif current_funding_condition:
                self.logger().info("Funding rate difference reached for stop loss, stopping executors")
                self.stopped_funding_arbitrages[token].append(funding_arbitrage_info)
                stop_executor_actions.extend([StopExecutorAction(executor_id=executor.id) for executor in executors])
                stop_tokens.append(token)
        # Remove the tokens that were stopped from the active arbitrages
        for t in stop_tokens:
            self.active_funding_arbitrages.pop(t, None)
            if not reversion_profit_condition:
                self.closed_funding_time[t] = self.current_timestamp
        return stop_executor_actions

    def did_complete_funding_payment(self, funding_payment_completed_event: FundingPaymentCompletedEvent):
        """
        Based on the funding payment event received, check if one of the active arbitrages matches to add the event
        to the list.
        """
        token = funding_payment_completed_event.trading_pair.split("-")[0]
        if token in self.active_funding_arbitrages:
            self.active_funding_arbitrages[token]["funding_payments"].append(funding_payment_completed_event)

    def get_position_executors_config(self, token, connector_1, connector_2, trade_side):
        trading_pair_1 = self.get_trading_pair_for_connector(token, connector_1)
        trading_pair_2 = self.get_trading_pair_for_connector(token, connector_2)
        price = self.market_data_provider.get_price_by_type(
            connector_name=connector_1,
            trading_pair=trading_pair_1,
            price_type=PriceType.MidPrice
        )
        origin_amount = position_amount = self.config.position_size_quote / price
        position_amount = min(self.connectors[connector_1].quantize_order_amount(trading_pair_1, position_amount),
                              self.connectors[connector_2].quantize_order_amount(trading_pair_2, position_amount))
        if position_amount <= 0:
            self.logger().warning(f"Position amount is zero or negative for {token} on {connector_1} and {connector_2}. Origin amount: {origin_amount}")
            return None, None

        position_executor_config_1 = PositionExecutorConfig(
            timestamp=self.current_timestamp,
            connector_name=connector_1,
            trading_pair=trading_pair_1,
            side=trade_side,
            amount=position_amount,
            leverage=self.config.leverage,
            triple_barrier_config=TripleBarrierConfig(open_order_type=OrderType.MARKET),
        )
        position_executor_config_2 = PositionExecutorConfig(
            timestamp=self.current_timestamp,
            connector_name=connector_2,
            trading_pair=trading_pair_2,
            side=TradeType.BUY if trade_side == TradeType.SELL else TradeType.SELL,
            amount=position_amount,
            leverage=self.config.leverage,
            triple_barrier_config=TripleBarrierConfig(open_order_type=OrderType.MARKET),
        )
        return position_executor_config_1, position_executor_config_2

    def format_status(self) -> str:
        original_status = super().format_status()
        funding_rate_status = []
        if self.ready_to_trade:
            all_funding_info = []
            payment_status = []
            for token, funding_arbitrage_info in self.active_funding_arbitrages.items():
                long_connector = funding_arbitrage_info["connector_1"] if funding_arbitrage_info["side"] == TradeType.BUY else funding_arbitrage_info["connector_2"]
                short_connector = funding_arbitrage_info["connector_2"] if funding_arbitrage_info["side"] == TradeType.BUY else funding_arbitrage_info["connector_1"]

                payments = funding_arbitrage_info["funding_payments"]
                payment_status.append({
                    'Token': token,
                    'Payments Count': len(payments),
                    'Amount': sum(payment.amount for payment in payments),
                    'Time': datetime.fromtimestamp(payments[-1].timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S') if payments else "N/A"
                })

                funding_info_report = self.get_funding_info_by_token(token)
                long_connector_funding_rate = self.get_normalized_funding_rate_in_seconds(funding_info_report, long_connector) * self.funding_profitability_interval
                short_connector_funding_rate = self.get_normalized_funding_rate_in_seconds(funding_info_report, short_connector) * self.funding_profitability_interval
                funding_rate_diff_in_day = short_connector_funding_rate - long_connector_funding_rate
                executors = self.filter_executors(
                    executors=self.get_all_executors(),
                    filter_func=lambda x: x.id in funding_arbitrage_info["executors_ids"]
                )
                trading_fee_pct = sum(executor.cum_fees_quote / executor.filled_amount_quote for executor in executors)
                days_to_profit = -trading_fee_pct / funding_rate_diff_in_day if funding_rate_diff_in_day != 0 else float('inf')
                days_to_take_profit = (self.config.profitability_to_take_profit - trading_fee_pct) / funding_rate_diff_in_day if funding_rate_diff_in_day != 0 else float('inf')
                close_position_net_pnl_pct = sum(
                    executor.net_pnl_pct - self.get_close_fee(
                        connector_name=executor.connector_name,
                        trading_pair=executor.trading_pair,
                        order_type=OrderType.MARKET,
                        order_side=executor.side,
                        amount=Decimal('0')
                    ) for executor in executors)
                time_to_next_funding_info_c1 = funding_info_report[long_connector].next_funding_utc_timestamp - self.current_timestamp
                time_to_next_funding_info_c2 = funding_info_report[short_connector].next_funding_utc_timestamp - self.current_timestamp
                funding_info = {
                    "Token": token,
                    "Long Connector": long_connector,
                    "Short Connector": short_connector,
                    "Long Funding Rate In Day": round(long_connector_funding_rate, 5),
                    "Short Funding Rate In Day": round(short_connector_funding_rate, 5),
                    "Funding Rate Diff In Day": round(funding_rate_diff_in_day, 5),
                    "Close Position Net PNL PCT": round(close_position_net_pnl_pct, 5),
                    "Days To Profit": days_to_profit,
                    "Days To Take Profit": days_to_take_profit,
                    "Min To Funding(long)": time_to_next_funding_info_c1 / 60,
                    "Min To Funding(short)": time_to_next_funding_info_c2 / 60
                }
                all_funding_info.append(funding_info)

            collected_payment_status = []
            for token, funding_arbitrage_info in self.stopped_funding_arbitrages.items():
                collected_payment_status.append({
                    'Token': token,
                    'Payments Count': sum(len(info["funding_payments"]) for info in funding_arbitrage_info),
                    'Amount': sum(payment.amount for info in funding_arbitrage_info for payment in info["funding_payments"]),
                })
            df = pd.DataFrame(collected_payment_status)
            sum_row = df.sum(numeric_only=True)
            collected_payment_status_df = pd.concat([df, pd.DataFrame([sum_row])])

            funding_rate_status.append(f"\n\n\nMin Funding Rate Profitability: {self.config.min_funding_rate_profitability:.2%}")
            funding_rate_status.append(f"Profitability to Take Profit: {self.config.profitability_to_take_profit:.2%}\n")
            funding_rate_status.append(format_df_for_printout(df=pd.DataFrame(all_funding_info), table_format="psql",))
            funding_rate_status.append("\n\nFunding Payments Status:")
            funding_rate_status.append(format_df_for_printout(df=pd.DataFrame(payment_status), table_format="psql",))
            funding_rate_status.append("\n\nCollected Funding Payments Status:")
            funding_rate_status.append(format_df_for_printout(df=collected_payment_status_df, table_format="psql",))

        return original_status + "\n".join(funding_rate_status)

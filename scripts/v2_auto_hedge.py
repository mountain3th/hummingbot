import os
from decimal import Decimal
from typing import Dict, List, Optional, Set, Union

import pandas as pd

from hummingbot.client.ui.interface_utils import format_df_for_printout
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.connector.utils import combine_to_hb_trading_pair
from hummingbot.core.clock import Clock
from hummingbot.core.data_type.common import OrderType, PositionAction, PositionMode, PositionSide, TradeType
from hummingbot.core.data_type.limit_order import LimitOrder
from hummingbot.core.data_type.order_candidate import OrderCandidate, PerpetualOrderCandidate
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.model.position import Position
from hummingbot.strategy.market_trading_pair_tuple import MarketTradingPairTuple
from hummingbot.strategy.strategy_v2_base import StrategyV2Base, StrategyV2ConfigBase
from hummingbot.strategy.utils import order_age
from hummingbot.strategy_v2.models.base import RunnableStatus
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, StopExecutorAction


class AutoHedgeV2StrategyConfig(StrategyV2ConfigBase):
    script_file_name: str = os.path.basename(__file__)
    candles_config: List[CandlesConfig] = []
    markets: Dict[str, Set[str]] = {}
    hedge_markets: Dict[str, str] = {}
    check_hedge_interval: int = 60  # seconds
    leverage: Optional[Decimal] = None
    min_trade_size: Optional[Decimal] = Decimal("5.0")


class AutoHedgeV2Strategy(StrategyV2Base):

    def __init__(self, connectors: Dict[str, ConnectorBase], config: AutoHedgeV2StrategyConfig):
        super().__init__(connectors, config)
        self.config = config
        self.closed_executors_buffer: int = 30
        self._leverage: Decimal = config.leverage if config.leverage else Decimal(3)
        self._position_mode = PositionMode.HEDGE
        self._slippage_multiplier = lambda is_buy: Decimal(1.02) if is_buy else Decimal(0.98)
        self._min_trade_size = config.min_trade_size if config.min_trade_size else Decimal("5.0")
        self._max_order_age = 300

    def start(self, clock: Clock, timestamp: float) -> None:
        """
        Start the strategy.
        :param clock: Clock to use.
        :param timestamp: Current time.
        """
        super().start(clock, timestamp)
        self._check_hedge_timestamp = timestamp

        self.hedge_market_pairs = {token: MarketTradingPairTuple(market=self.connectors[exchange_name], trading_pair=combine_to_hb_trading_pair(token, 'USDT'), base_asset=token, quote_asset='USDT')
                                   for token, exchange_name in self.config.hedge_markets.items()
                                   if combine_to_hb_trading_pair(token, 'USDT') in self.connectors[exchange_name].trading_pairs}

    def on_tick(self):
        super().on_tick()
        self.check_and_cancel_active_orders()
        self.check_manual_kill_switch()
        if self.current_timestamp - self._check_hedge_timestamp > self.config.check_hedge_interval:
            self.control_hedge()

    def control_hedge(self):
        self._check_hedge_timestamp = self.current_timestamp
        if not self.hedge_market_pairs:
            return

        need_hedge_tokens = self.get_tokens_to_hedge()

        df = self.get_balance_df()
        df = df[~df["Exchange"].apply(self.is_perpetual)]
        df = df[df["Asset"].isin(need_hedge_tokens)]
        df = df.groupby('Asset').sum().reset_index()

        def hedge(row):
            asset, amount = row["Asset"], Decimal(row["Total Balance"])
            market_pair = self.hedge_market_pairs.get(asset)
            positions = self.get_positions(market_pair)
            hedge_amount = sum(map(lambda position: position.amount if position.position_side in [PositionSide.LONG, PositionSide.BOTH] else -abs(position.amount), positions))
            net_amount = hedge_amount + amount
            is_buy = net_amount < 0
            amount_to_hedge = abs(net_amount)
            if amount_to_hedge == 0:
                self.logger().info(f"No hedge needed for {asset}: current balance {amount}, net hedge amount {hedge_amount}")
                return
            price = self.hedge_market_pairs[asset].get_mid_price() * self._slippage_multiplier(is_buy=is_buy)
            self.logger().info(f"Hedging {asset}: current balance {amount}, hedge amount {hedge_amount}, net hedge amount {net_amount}, amount to hedge {amount_to_hedge} at price {price}")
            order_candidates = self.get_perpetual_order_candidates(
                market_pair=market_pair,
                is_buy=is_buy,
                amount=amount_to_hedge,
                price=price
            )
            if not order_candidates:
                self.logger().warning(f"No valid order candidates for hedging {asset}. Skipping.")
                return
            self.place_orders(market_pair=market_pair, orders=order_candidates)

        df.apply(hedge, axis=1)

    def place_orders(
        self, market_pair: MarketTradingPairTuple, orders: Union[List[OrderCandidate], List[PerpetualOrderCandidate]]
    ) -> None:
        """
        Place an order refering the order candidates.
        :params market_pair: The market pair to place the order.
        :params orders: The list of orders to place.
        """
        for order in orders:
            self.logger().info(f"Create {order.order_side} {order.amount} {order.trading_pair} at {order.price}")
            is_buy = order.order_side == TradeType.BUY
            amount = order.amount
            price = order.price
            position_action = PositionAction.OPEN
            if isinstance(order, PerpetualOrderCandidate) and order.position_close:
                position_action = PositionAction.CLOSE
            trade = self.buy_with_specific_market if is_buy else self.sell_with_specific_market
            trade(market_pair, amount=amount, order_type=OrderType.LIMIT, price=price, position_action=position_action)

    def create_actions_proposal(self) -> List[CreateExecutorAction]:
        return []

    def stop_actions_proposal(self) -> List[StopExecutorAction]:
        return []

    def apply_initial_setting(self):
        connectors_position_mode = {}
        for controller_id, controller in self.controllers.items():
            config_dict = controller.config.dict()
            if "connector_name" in config_dict:
                if self.is_perpetual(config_dict["connector_name"]):
                    if "position_mode" in config_dict:
                        connectors_position_mode[config_dict["connector_name"]] = config_dict["position_mode"]
                    if "leverage" in config_dict:
                        self.connectors[config_dict["connector_name"]].set_leverage(leverage=config_dict["leverage"],
                                                                                    trading_pair=config_dict["trading_pair"])
        for connector_name, position_mode in connectors_position_mode.items():
            self.connectors[connector_name].set_position_mode(position_mode)
        for connector_name in self.config.hedge_markets.values():
            self.connectors[connector_name].set_position_mode(self._position_mode)

    def get_positions(self, market_pair: MarketTradingPairTuple, position_side: PositionSide = None) -> List[Position]:
        """
        Get the active positions of a market.
        :param market_pair: Market pair to get the positions of.
        :return: The active positions of the market.
        """
        trading_pair = market_pair.trading_pair
        positions: List[Position] = [
            position
            for position in market_pair.market.account_positions.values()
            if not isinstance(position, PositionMode) and position.trading_pair == trading_pair
        ]
        if position_side:
            return [position for position in positions if position.position_side == position_side]
        return positions

    def get_perpetual_order_candidates(
        self, market_pair: MarketTradingPairTuple, is_buy: bool, amount: Decimal, price: Decimal
    ) -> List[PerpetualOrderCandidate]:
        """
        Check if the balance is sufficient to place an order.
        if not, adjust the amount to the balance available.
        returns the order candidate if the order meets the accepted criteria
        else, return None
        """
        def get_closing_order_candidate(is_buy: bool, amount: Decimal, price: Decimal) -> Union[PerpetualOrderCandidate, None]:
            opp_position_side = PositionSide.SHORT if is_buy else PositionSide.LONG
            opp_position_list = self.get_positions(market_pair, opp_position_side)
            # opp_position_list should only have 1 position
            for opp_position in opp_position_list:
                close_amount = min(amount, abs(opp_position.amount))
                order_candidate = PerpetualOrderCandidate(
                    trading_pair=market_pair.trading_pair,
                    is_maker=False,
                    order_side=TradeType.BUY if is_buy else TradeType.SELL,
                    amount=close_amount,
                    price=price,
                    order_type=OrderType.LIMIT,
                    leverage=Decimal(self._leverage),
                    position_close=True,
                )
                adjusted_candidate_order = budget_checker.adjust_candidate(order_candidate, all_or_none=False)
                return adjusted_candidate_order
            return None

        budget_checker = market_pair.market.budget_checker
        if amount * price < self._min_trade_size:
            self.logger().info("trade value (%s) is less than min trade size. (%s)", amount * price, self._min_trade_size)
            return []
        order_candidates = []
        if self._position_mode == PositionMode.HEDGE:
            order_candidate = get_closing_order_candidate(is_buy, amount, price)
            if order_candidate:
                order_candidates.append(order_candidate)
                amount -= order_candidate.amount
        order_candidate = PerpetualOrderCandidate(
            trading_pair=market_pair.trading_pair,
            is_maker=False,
            order_type=OrderType.LIMIT,
            order_side=TradeType.BUY if is_buy else TradeType.SELL,
            amount=amount,
            price=price,
            leverage=Decimal(self._leverage),
        )
        adjusted_candidate_order = budget_checker.adjust_candidate(order_candidate, all_or_none=False)
        if adjusted_candidate_order.amount > 0:
            order_candidates.append(adjusted_candidate_order)
        return order_candidates

    def get_tokens_to_hedge(self) -> bool:
        """
        Check if there are any active orders and cancel them
        :return: True if there are active orders, False otherwise.
        """
        results = []
        for token, connector_name in self.config.hedge_markets.items():
            results.append(token)
            for order in self.get_active_orders(connector_name):
                if self.is_hedge_order(order):
                    results.remove(token)
        return results

    def is_hedge_order(self, order: LimitOrder) -> bool:
        market_pair: MarketTradingPairTuple = self.order_tracker.get_market_pair_from_order_id(order.client_order_id)
        return market_pair in self.hedge_market_pairs.values()

    def check_and_cancel_active_orders(self):
        for connector_name in self.config.hedge_markets.values():
            for order in self.get_active_orders(connector_name):
                if not self.is_hedge_order(order):
                    continue
                if order_age(order, self.current_timestamp) < self._max_order_age:
                    continue
                self.logger().info(
                    f"Cancel {'buy' if order.trade_type == TradeType.BUY else 'sell'} {order.amount} {order.trading_pair} at {order.price}"
                )
                self.cancel(order, order.trading_pair, order.client_order_id)

    def check_manual_kill_switch(self):
        if self._is_stop_triggered:
            return
        for controller_id, controller in self.controllers.items():
            if controller.config.manual_kill_switch and controller.status == RunnableStatus.RUNNING:
                self.logger().info(f"Manual cash out for controller {controller_id}.")
                controller.stop()
                executors_to_stop = self.get_executors_by_controller(controller_id)
                self.executor_orchestrator.execute_actions(
                    [StopExecutorAction(executor_id=executor.id,
                                        controller_id=executor.controller_id) for executor in executors_to_stop])
            if not controller.config.manual_kill_switch and controller.status == RunnableStatus.TERMINATED:
                if controller_id in self.drawdown_exited_controllers:
                    continue
                self.logger().info(f"Restarting controller {controller_id}.")
                controller.start()

    def format_status(self) -> str:
        status = super().format_status()

        lines = ["\n"]
        lines += ["Auto Hedge Strategy Status:"]

        df = self.get_balance_df()
        df = df[~df["Exchange"].apply(self.is_perpetual)]
        df = df.groupby('Asset').sum().reset_index()

        hedge_infos = []
        for _, market_pair in self.hedge_market_pairs.items():
            positions = self.get_positions(market_pair)
            hedge_amount = sum(map(lambda position: position.amount if position.position_side in [PositionSide.LONG, PositionSide.BOTH] else -abs(position.amount), positions))
            balance_amount = Decimal(df[df["Asset"] == market_pair.base_asset]["Total Balance"].values[0]) if not df[df["Asset"] == market_pair.base_asset].empty else Decimal("0.0")
            hedge_infos.append({
                "Hedge Pair": market_pair.trading_pair,
                "Balance Amount": balance_amount,
                "Hedge Amount": hedge_amount,
                "Net Amount": hedge_amount + balance_amount,
                "Leverage": self._leverage,
            })

        lines.append(format_df_for_printout(pd.DataFrame(hedge_infos), table_format="psql"))

        return status + "\n".join(lines)

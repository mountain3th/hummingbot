import time
from decimal import Decimal
from typing import Dict, List, Set

import pandas as pd
import pandas_ta as ta
from pydantic import Field, field_validator

from hummingbot.client.ui.interface_utils import format_df_for_printout
from hummingbot.core.data_type.common import OrderType, PriceType, TradeType
from hummingbot.core.utils.estimate_fee import build_trade_fee
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.controllers.controller_base import ControllerBase, ControllerConfigBase
from hummingbot.strategy_v2.executors.data_types import ConnectorPair
from hummingbot.strategy_v2.executors.xemm_executor.data_types import XEMMExecutorConfig
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, ExecutorAction


class XEMMMultipleLevelsConfig(ControllerConfigBase):
    controller_name: str = "xemm_multiple_levels"
    candles_config: List[CandlesConfig] = []
    maker_connector: str = Field(
        default="mexc",
        json_schema_extra={"prompt": "Enter the maker connector: ", "prompt_on_new": True})
    maker_trading_pair: str = Field(
        default="PEPE-USDT",
        json_schema_extra={"prompt": "Enter the maker trading pair: ", "prompt_on_new": True})
    taker_connector: str = Field(
        default="binance",
        json_schema_extra={"prompt": "Enter the taker connector: ", "prompt_on_new": True})
    taker_trading_pair: str = Field(
        default="PEPE-USDT",
        json_schema_extra={"prompt": "Enter the taker trading pair: ", "prompt_on_new": True})
    buy_levels_targets_amount: List[List[Decimal]] = Field(
        default="0.003,10-0.006,20-0.009,30",
        json_schema_extra={
            "prompt": "Enter the buy levels targets with the following structure: (target_profitability1,amount1-target_profitability2,amount2): ",
            "prompt_on_new": True,
            "is_updatable": True})
    sell_levels_targets_amount: List[List[Decimal]] = Field(
        default="0.003,10-0.006,20-0.009,30",
        json_schema_extra={
            "prompt": "Enter the sell levels targets with the following structure: (target_profitability1,amount1-target_profitability2,amount2): ",
            "prompt_on_new": True,
            "is_updatable": True})
    min_profitability: Decimal = Field(
        default=0.003,
        json_schema_extra={"prompt": "Enter the minimum profitability: ", "prompt_on_new": True, "is_updatable": True})
    max_profitability: Decimal = Field(
        default=0.01,
        json_schema_extra={"prompt": "Enter the maximum profitability: ", "prompt_on_new": True, "is_updatable": True})
    max_executors_imbalance: int = Field(
        default=1,
        json_schema_extra={"prompt": "Enter the maximum executors imbalance: ", "prompt_on_new": True})
    max_reenter_delay: int = Field(
        default=600,
        json_schema_extra={"prompt": "Enter the max delay to execute next run: ", "prompt_on_new": True})

    @field_validator("buy_levels_targets_amount", "sell_levels_targets_amount", mode="before")
    @classmethod
    def validate_levels_targets_amount(cls, v):
        if isinstance(v, str):
            v = [list(map(Decimal, x.split(","))) for x in v.split("-")]
        return v

    def update_markets(self, markets: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
        if self.maker_connector not in markets:
            markets[self.maker_connector] = set()
        markets[self.maker_connector].add(self.maker_trading_pair)
        if self.taker_connector not in markets:
            markets[self.taker_connector] = set()
        markets[self.taker_connector].add(self.taker_trading_pair)
        return markets


class XEMMMultipleLevels(ControllerBase):

    def __init__(self, config: XEMMMultipleLevelsConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.consecute_buy_fails = 0
        self.consecute_sell_fails = 0
        self.enter_buy_recover_timestamp = 0
        self.enter_sell_recover_timestamp = 0
        self.fee_update_tick = 0
        self.maker_fee = None
        self.taker_fee = None
        self.max_records = 200
        self.volatility_interval = 60
        self.volatility_data = None
        self.candles = self.market_data_provider.get_candles_feed(CandlesConfig(connector=self.config.maker_connector,
                                                                                trading_pair=self.config.maker_trading_pair,
                                                                                interval='15m',
                                                                                max_records=self.max_records))

    def get_wait_time(self, fail_count):
        base_delay = 5
        max_delay = self.config.max_reenter_delay
        return min(base_delay * (2**fail_count), max_delay)

    async def update_processed_data(self):
        await self.candles.fill_historical_candles()
        df = self.candles.candles_df
        df["volatility"] = df["close"].pct_change().rolling(self.volatility_interval).std()
        df["volatility_pct"] = df["volatility"] / df["close"]
        df["volatility_pct_mean"] = df["volatility_pct"].rolling(self.volatility_interval).mean()
        df["natr"] = ta.natr(df["high"], df["low"], df["close"], length=self.volatility_interval)
        self.volatility_data = df.iloc[-1]
        if self.fee_update_tick % 3600 == 0:
            self.fee_update_tick = 0
            self.maker_fee = build_trade_fee(self.config.maker_connector, True, base_currency='', quote_currency='', order_type=OrderType.LIMIT,
                                             order_side=TradeType.BUY, amount=Decimal('0'))
            self.taker_fee = build_trade_fee(self.config.taker_connector, False, base_currency='', quote_currency='', order_type=OrderType.MARKET,
                                             order_side=TradeType.BUY, amount=Decimal('0'))
        self.fee_update_tick += 1

    def determine_executor_actions(self) -> List[ExecutorAction]:
        current_timestamp = int(time.time())
        executor_actions = []
        mid_price = self.market_data_provider.get_price_by_type(self.config.maker_connector, self.config.maker_trading_pair, PriceType.MidPrice)
        maker_best_bid = self.market_data_provider.get_price_by_type(self.config.maker_connector, self.config.maker_trading_pair, PriceType.BestBid)
        maker_best_ask = self.market_data_provider.get_price_by_type(self.config.maker_connector, self.config.maker_trading_pair, PriceType.BestAsk)
        taker_bid_price = self.market_data_provider.get_price_for_quote_volume(self.config.taker_connector, self.config.taker_trading_pair, self.config.buy_levels_targets_amount[0][1], is_buy=False).result_price
        taker_ask_price = self.market_data_provider.get_price_for_quote_volume(self.config.taker_connector, self.config.taker_trading_pair, self.config.sell_levels_targets_amount[0][1], is_buy=True).result_price

        enter_buy, enter_sell = True, True
        maker_bid_price = Decimal(taker_bid_price) * (1 - self.config.buy_levels_targets_amount[0][0] - self.maker_fee.percent - self.taker_fee.percent)
        maker_ask_price = Decimal(taker_ask_price) * (1 + self.config.sell_levels_targets_amount[0][0] + self.maker_fee.percent + self.taker_fee.percent)
        if maker_bid_price >= maker_best_ask:
            enter_buy = False
        if maker_ask_price <= maker_best_bid:
            enter_sell = False

        active_buy_executors = self.filter_executors(
            executors=self.executors_info,
            filter_func=lambda e: not e.is_done and e.config.maker_side == TradeType.BUY
        )
        active_sell_executors = self.filter_executors(
            executors=self.executors_info,
            filter_func=lambda e: not e.is_done and e.config.maker_side == TradeType.SELL
        )
        completed_buy_executors = self.filter_executors(
            executors=self.executors_info,
            filter_func=lambda e: e.is_done and e.config.maker_side == TradeType.BUY and e.filled_amount_quote != 0
        )
        completed_sell_executors = self.filter_executors(
            executors=self.executors_info,
            filter_func=lambda e: e.is_done and e.config.maker_side == TradeType.SELL and e.filled_amount_quote != 0
        )
        stopped_buy_executors = self.filter_executors(
            executors=self.executors_info[-10:],
            filter_func=lambda e: e.is_done and e.config.maker_side == TradeType.BUY and e.filled_amount_quote == 0
        )
        if stopped_buy_executors and self.enter_buy_recover_timestamp <= stopped_buy_executors[-1].close_timestamp:
            self.enter_buy_recover_timestamp = current_timestamp + self.get_wait_time(self.consecute_buy_fails)
            self.logger().info(f"Backoff for buy market {self.config.maker_trading_pair}. Wait for {self.get_wait_time(self.consecute_buy_fails)}s")
            self.consecute_buy_fails += 1
        elif active_buy_executors and self.enter_buy_recover_timestamp <= active_buy_executors[-1].timestamp:
            self.consecute_buy_fails = 0
        stopped_sell_executors = self.filter_executors(
            executors=self.executors_info[-10:],
            filter_func=lambda e: e.is_done and e.config.maker_side == TradeType.SELL and e.filled_amount_quote == 0
        )
        if stopped_sell_executors and self.enter_sell_recover_timestamp <= stopped_sell_executors[-1].close_timestamp:
            self.enter_sell_recover_timestamp = current_timestamp + self.get_wait_time(self.consecute_sell_fails)
            self.logger().info(f"Backoff for sell market {self.config.maker_trading_pair}. Wait for {self.get_wait_time(self.consecute_sell_fails)}s")
            self.consecute_sell_fails += 1
        elif active_sell_executors and self.enter_sell_recover_timestamp <= active_sell_executors[-1].timestamp:
            self.consecute_sell_fails = 0

        if self.enter_buy_recover_timestamp >= current_timestamp:
            enter_buy = False
        if self.enter_sell_recover_timestamp >= current_timestamp:
            enter_sell = False

        imbalance = len(completed_buy_executors) - len(completed_sell_executors)

        if enter_buy:
            for target_profitability, amount in self.config.buy_levels_targets_amount:
                active_buy_executors_target = [e.config.target_profitability == target_profitability for e in active_buy_executors]

                if len(active_buy_executors_target) == 0 and imbalance < self.config.max_executors_imbalance:
                    min_profitability = target_profitability - self.config.min_profitability
                    max_profitability = target_profitability + self.config.max_profitability
                    config = XEMMExecutorConfig(
                        controller_id=self.config.id,
                        timestamp=self.market_data_provider.time(),
                        buying_market=ConnectorPair(connector_name=self.config.maker_connector,
                                                    trading_pair=self.config.maker_trading_pair),
                        selling_market=ConnectorPair(connector_name=self.config.taker_connector,
                                                     trading_pair=self.config.taker_trading_pair),
                        maker_side=TradeType.BUY,
                        order_amount=amount / mid_price,
                        min_profitability=min_profitability,
                        target_profitability=target_profitability,
                        max_profitability=max_profitability
                    )
                    executor_actions.append(CreateExecutorAction(executor_config=config, controller_id=self.config.id))
        if enter_sell:
            for target_profitability, amount in self.config.sell_levels_targets_amount:
                active_sell_executors_target = [e.config.target_profitability == target_profitability for e in active_sell_executors]
                if len(active_sell_executors_target) == 0 and imbalance > -self.config.max_executors_imbalance:
                    min_profitability = target_profitability - self.config.min_profitability
                    max_profitability = target_profitability + self.config.max_profitability
                    config = XEMMExecutorConfig(
                        controller_id=self.config.id,
                        timestamp=time.time(),
                        buying_market=ConnectorPair(connector_name=self.config.taker_connector,
                                                    trading_pair=self.config.taker_trading_pair),
                        selling_market=ConnectorPair(connector_name=self.config.maker_connector,
                                                     trading_pair=self.config.maker_trading_pair),
                        maker_side=TradeType.SELL,
                        order_amount=amount / mid_price,
                        min_profitability=min_profitability,
                        target_profitability=target_profitability,
                        max_profitability=max_profitability
                    )
                    executor_actions.append(CreateExecutorAction(executor_config=config, controller_id=self.config.id))
        return executor_actions

    def to_format_status(self) -> List[str]:
        lines = []
        if self.volatility_data is not None:
            lines.append(format_df_for_printout(pd.DataFrame(self.volatility_data).T, table_format="psql", ))

        all_executors_custom_info = pd.DataFrame(e.custom_info for e in self.executors_info if e.is_active)
        lines.extend([format_df_for_printout(all_executors_custom_info, table_format="psql", )])

        return lines

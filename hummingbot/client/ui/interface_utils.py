import asyncio
import itertools
from datetime import datetime, time, timedelta
from decimal import Decimal
from typing import List, Optional, Set, Tuple

import pandas as pd
import psutil
import tabulate
from yarl import Query

from hummingbot.client.config.config_data_types import ClientConfigEnum
from hummingbot.client.performance import PerformanceMetrics
from hummingbot.core.utils.mail_tool import create_email, send_email
from hummingbot.model.executors import Executors
from hummingbot.model.trade_fill import TradeFill
from hummingbot.strategy_v2.models.executors import CloseType

s_decimal_0 = Decimal("0")


def format_bytes(size):
    for unit in ["B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB"]:
        if abs(size) < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} YB"


async def start_timer(timer):
    count = 1
    while True:
        count += 1

        mins, sec = divmod(count, 60)
        hour, mins = divmod(mins, 60)
        days, hour = divmod(hour, 24)

        timer.log(f"Uptime: {days:>3} day(s), {hour:02}:{mins:02}:{sec:02}")
        await _sleep(1)


async def _sleep(delay):
    """
    A wrapper function that facilitates patching the sleep in unit tests without affecting the asyncio module
    """
    await asyncio.sleep(delay)


async def start_process_monitor(process_monitor):
    hb_process = psutil.Process()
    while True:
        with hb_process.oneshot():
            threads = hb_process.num_threads()
            process_monitor.log("CPU: {:>5}%, ".format(hb_process.cpu_percent()) +
                                "Mem: {:>10} ({}), ".format(
                                    format_bytes(hb_process.memory_info().vms / threads),
                                    format_bytes(hb_process.memory_info().rss)) +
                                "Threads: {:>3}, ".format(threads)
                                )
        await _sleep(1)


async def start_trade_monitor(trade_monitor):
    from hummingbot.client.hummingbot_application import HummingbotApplication
    hb = HummingbotApplication.main_application()
    trade_monitor.log("Trades: 0, Total P&L: 0.00, Return %: 0.00%")

    while True:
        try:
            if hb.trading_core._strategy_running and hb.trading_core.strategy is not None:
                if all(market.ready for market in hb.trading_core.markets.values()):
                    with hb.trading_core.trade_fill_db.get_new_session() as session:
                        trades: List[TradeFill] = hb._get_trades_from_session(
                            int(hb.init_time * 1e3),
                            session=session,
                            config_file_path=hb.strategy_file_name)
                        if len(trades) > 0:
                            return_pcts = []
                            pnls = []
                            market_info: Set[Tuple[str, str]] = set((t.market, t.symbol) for t in trades)
                            for market, symbol in market_info:
                                cur_trades = [t for t in trades if t.market == market and t.symbol == symbol]
                                cur_balances = await hb.trading_core.get_current_balances(market)
                                perf = await PerformanceMetrics.create(symbol, cur_trades, cur_balances)
                                return_pcts.append(perf.return_pct)
                                pnls.append(perf.total_pnl)
                            avg_return = sum(return_pcts) / len(return_pcts) if len(return_pcts) > 0 else s_decimal_0
                            quote_assets = set(t.symbol.split("-")[1] for t in trades)
                            if len(quote_assets) == 1:
                                total_pnls = f"{PerformanceMetrics.smart_round(sum(pnls))} {list(quote_assets)[0]}"
                            else:
                                total_pnls = "N/A"
                            trade_monitor.log(f"Trades: {len(trades)}, Total P&L: {total_pnls}, "
                                              f"Return %: {avg_return:.2%}")
            await _sleep(2.0)  # sleeping for longer to manage resources
        except asyncio.CancelledError:
            raise
        except Exception:
            hb.logger().exception("start_trade_monitor failed.")
            await _sleep(2.0)


async def performance_send(force=False):
    from hummingbot.client.hummingbot_application import HummingbotApplication
    hb = HummingbotApplication.main_application()

    async def report():
        markets = []
        pnls = []
        trades_count = []
        now = datetime.now()
        now = now.replace(hour=21, minute=0, second=0, microsecond=0)
        start_time = (now - timedelta(days=1)).timestamp()
        end_time = now.timestamp()

        hb.logger().info(f"Generating performance report for {now.strftime('%Y-%m-%d')}...")
        try:
            if (task := hb.trading_core.strategy_task) and not task.done():
                if all(market.ready for market in hb.markets.values()):
                    with hb.trade_fill_db.get_new_session() as session:
                        filters = [Executors.close_timestamp >= start_time,
                                   Executors.close_timestamp < end_time,
                                   Executors.close_type == CloseType.COMPLETED.value]
                        query: Query = (session
                                        .query(Executors)
                                        .filter(*filters)
                                        .order_by(Executors.timestamp.desc()))

                        results: List[Executors] = query.all() or []

                        for key, group in itertools.groupby(results, key=lambda x: x.controller_id):
                            group = list(group)
                            markets.append(key)
                            trades_count.append(len(group))
                            pnls.append(sum(executor.net_pnl_quote for executor in group))

                        df = pd.DataFrame({
                            "Market": markets,
                            "Trades": trades_count,
                            "Total P&L": pnls,
                        })
                        summary = df.sum(numeric_only=True)
                        summary['Market'] = '-'
                        df = pd.concat([df, summary.to_frame().T])

                        df.to_html("performance_report.html", index=True)
                        # df_str = format_df_for_printout(df, hb.client_config_map.tables_format)
                        with open("performance_report.html") as f:
                            html = f.read()
                        email_message = create_email(
                            subject=f"Daily Performance Report - {now.strftime('%Y-%m-%d')}",
                            recipients=hb.client_config_map.email_recipients,
                            body=html,
                            body_type="html")
                        send_email(message=email_message)
                        hb.logger().info(f"Performance report for {now.strftime('%Y-%m-%d')} generated successfully.")
        except asyncio.CancelledError:
            raise
        except Exception:
            hb.logger().exception("performance send report failed.")

    if not force:
        while True:
            now = datetime.now()
            target_time = time(21, 0)  # 21:00
            # If it's not 21:00 yet today, skip
            if now.time() < target_time:
                next_call_time = datetime.combine(now.date(), target_time) - now
            elif (now - datetime.combine(now.date(), target_time)) > timedelta(hours=1):
                next_call_time = now - datetime.combine(now.date(), target_time)
            else:
                await report()
                next_call_time = timedelta(days=1)

            hb.logger().info(f"Performance report generating will be run after {next_call_time.total_seconds()}s")
            await asyncio.sleep(next_call_time.total_seconds())

    else:
        await report()


def format_df_for_printout(
    df: pd.DataFrame, table_format: ClientConfigEnum, max_col_width: Optional[int] = None, index: bool = False
) -> str:
    if max_col_width is not None:  # in anticipation of the next release of tabulate which will include maxcolwidth
        max_col_width = max(max_col_width, 4)
        df = df.astype(str).apply(
            lambda s: s.apply(
                lambda e: e if len(e) < max_col_width else f"{e[:max_col_width - 3]}..."
            )
        )
        df.columns = [c if len(c) < max_col_width else f"{c[:max_col_width - 3]}..." for c in df.columns]

    original_preserve_whitespace = tabulate.PRESERVE_WHITESPACE
    tabulate.PRESERVE_WHITESPACE = True
    try:
        formatted_df = tabulate.tabulate(df, tablefmt=table_format, showindex=index, headers="keys")
    finally:
        tabulate.PRESERVE_WHITESPACE = original_preserve_whitespace
    return formatted_df

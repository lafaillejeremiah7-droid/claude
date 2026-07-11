"""
State Machine Engine for XAUUSD Trading Bot

States:
    IDLE          → Bot inactive (dead zone, weekend, daily limit hit)
    SESSION_ANALYSIS → Evaluating market conditions at session open
    RANGE_MODE    → Asian session mean-reversion strategy active
    TREND_MODE    → London/NY trend-following strategy active
    NEWS_AVOID    → High-impact news blackout, no new entries
    ENTRY_SEARCH  → Actively scanning for entry signals
    POSITION_ACTIVE → Managing an open trade
    COOLDOWN      → Waiting period after trade close

Transitions are guarded by conditions that must be met before state change.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Callable, Optional
from datetime import datetime


class State(Enum):
    IDLE = "IDLE"
    SESSION_ANALYSIS = "SESSION_ANALYSIS"
    RANGE_MODE = "RANGE_MODE"
    TREND_MODE = "TREND_MODE"
    NEWS_AVOID = "NEWS_AVOID"
    ENTRY_SEARCH = "ENTRY_SEARCH"
    POSITION_ACTIVE = "POSITION_ACTIVE"
    COOLDOWN = "COOLDOWN"


@dataclass
class Transition:
    """A valid state transition with an optional guard condition."""
    from_state: State
    to_state: State
    guard: Optional[Callable[["BotContext"], bool]] = None
    description: str = ""


@dataclass
class BotContext:
    """Shared context passed through the state machine for guard evaluation."""
    current_time: datetime = field(default_factory=datetime.utcnow)
    session_mode: str = "IDLE"          # From session manager
    has_open_position: bool = False
    daily_loss_pct: float = 0.0
    daily_trade_count: int = 0
    session_trade_count: int = 0
    news_blackout_active: bool = False
    entry_score: float = 0.0
    cooldown_expires: Optional[datetime] = None
    last_trade_result: Optional[str] = None  # "WIN", "LOSS", or None
    spread_ok: bool = True
    max_daily_loss: float = 0.03
    max_trades_per_day: int = 6
    max_trades_per_session: int = 3
    entry_threshold: float = 0.70


class StateMachine:
    """
    Core state machine with explicit transitions and guard conditions.
    Only allows transitions that are registered and whose guards pass.
    """

    def __init__(self):
        self.state: State = State.IDLE
        self.transitions: list[Transition] = []
        self.history: list[tuple[datetime, State, State]] = []
        self._register_transitions()

    def _register_transitions(self):
        """Register all valid state transitions with their guard conditions."""

        # IDLE → SESSION_ANALYSIS: When a trading session opens
        self._add(State.IDLE, State.SESSION_ANALYSIS,
                  guard=self._guard_session_active,
                  description="Trading session has opened")

        # SESSION_ANALYSIS → RANGE_MODE: Asian session detected
        self._add(State.SESSION_ANALYSIS, State.RANGE_MODE,
                  guard=self._guard_is_range_session,
                  description="Asian session → range strategy")

        # SESSION_ANALYSIS → TREND_MODE: London/NY session detected
        self._add(State.SESSION_ANALYSIS, State.TREND_MODE,
                  guard=self._guard_is_trend_session,
                  description="London/NY session → trend strategy")

        # SESSION_ANALYSIS → IDLE: Dead zone or limits hit
        self._add(State.SESSION_ANALYSIS, State.IDLE,
                  guard=self._guard_should_idle,
                  description="No valid session or limits reached")

        # RANGE_MODE → ENTRY_SEARCH: Conditions met for scanning
        self._add(State.RANGE_MODE, State.ENTRY_SEARCH,
                  guard=self._guard_can_search_entry,
                  description="Range conditions valid, scanning entries")

        # TREND_MODE → ENTRY_SEARCH: Conditions met for scanning
        self._add(State.TREND_MODE, State.ENTRY_SEARCH,
                  guard=self._guard_can_search_entry,
                  description="Trend conditions valid, scanning entries")

        # ANY active mode → NEWS_AVOID: News blackout triggered
        for state in [State.RANGE_MODE, State.TREND_MODE, State.ENTRY_SEARCH]:
            self._add(state, State.NEWS_AVOID,
                      guard=self._guard_news_blackout,
                      description="High-impact news approaching")

        # NEWS_AVOID → SESSION_ANALYSIS: Blackout ended, re-evaluate
        self._add(State.NEWS_AVOID, State.SESSION_ANALYSIS,
                  guard=self._guard_news_clear,
                  description="News blackout ended, re-analyzing")

        # ENTRY_SEARCH → POSITION_ACTIVE: Entry signal confirmed
        self._add(State.ENTRY_SEARCH, State.POSITION_ACTIVE,
                  guard=self._guard_entry_confirmed,
                  description="Entry score >= threshold, opening position")

        # ENTRY_SEARCH → RANGE_MODE/TREND_MODE: No signal, return
        self._add(State.ENTRY_SEARCH, State.RANGE_MODE,
                  guard=self._guard_is_range_session,
                  description="No entry found, back to range scanning")
        self._add(State.ENTRY_SEARCH, State.TREND_MODE,
                  guard=self._guard_is_trend_session,
                  description="No entry found, back to trend scanning")

        # POSITION_ACTIVE → COOLDOWN: Trade closed
        self._add(State.POSITION_ACTIVE, State.COOLDOWN,
                  guard=self._guard_position_closed,
                  description="Position closed, entering cooldown")

        # POSITION_ACTIVE → IDLE: Daily limit hit while in trade
        self._add(State.POSITION_ACTIVE, State.IDLE,
                  guard=self._guard_daily_limit_hit,
                  description="Daily loss limit reached, shutting down")

        # COOLDOWN → SESSION_ANALYSIS: Cooldown expired
        self._add(State.COOLDOWN, State.SESSION_ANALYSIS,
                  guard=self._guard_cooldown_expired,
                  description="Cooldown complete, re-analyzing session")

        # COOLDOWN → IDLE: Session ended during cooldown or daily limit
        self._add(State.COOLDOWN, State.IDLE,
                  guard=self._guard_should_idle,
                  description="Session ended or limits hit during cooldown")

        # Any state → IDLE: Emergency / session end
        for state in State:
            if state != State.IDLE:
                self._add(state, State.IDLE,
                          guard=self._guard_emergency_stop,
                          description="Emergency stop or session end")

    def _add(self, from_state: State, to_state: State,
             guard: Optional[Callable] = None, description: str = ""):
        self.transitions.append(Transition(from_state, to_state, guard, description))

    # ------------------------------------------------------------------
    # GUARD CONDITIONS
    # ------------------------------------------------------------------

    def _guard_session_active(self, ctx: BotContext) -> bool:
        return ctx.session_mode != "IDLE" and ctx.daily_loss_pct < ctx.max_daily_loss

    def _guard_is_range_session(self, ctx: BotContext) -> bool:
        return ctx.session_mode == "RANGE"

    def _guard_is_trend_session(self, ctx: BotContext) -> bool:
        return ctx.session_mode in ("TREND", "TREND_AGGRESSIVE")

    def _guard_should_idle(self, ctx: BotContext) -> bool:
        return (ctx.session_mode == "IDLE" or
                ctx.daily_loss_pct >= ctx.max_daily_loss or
                ctx.daily_trade_count >= ctx.max_trades_per_day)

    def _guard_can_search_entry(self, ctx: BotContext) -> bool:
        return (not ctx.news_blackout_active and
                not ctx.has_open_position and
                ctx.spread_ok and
                ctx.session_trade_count < ctx.max_trades_per_session and
                ctx.daily_trade_count < ctx.max_trades_per_day and
                ctx.daily_loss_pct < ctx.max_daily_loss)

    def _guard_news_blackout(self, ctx: BotContext) -> bool:
        return ctx.news_blackout_active

    def _guard_news_clear(self, ctx: BotContext) -> bool:
        return not ctx.news_blackout_active

    def _guard_entry_confirmed(self, ctx: BotContext) -> bool:
        return ctx.entry_score >= ctx.entry_threshold and ctx.spread_ok

    def _guard_position_closed(self, ctx: BotContext) -> bool:
        return not ctx.has_open_position and ctx.last_trade_result is not None

    def _guard_daily_limit_hit(self, ctx: BotContext) -> bool:
        return ctx.daily_loss_pct >= ctx.max_daily_loss

    def _guard_cooldown_expired(self, ctx: BotContext) -> bool:
        if ctx.cooldown_expires is None:
            return True
        return ctx.current_time >= ctx.cooldown_expires

    def _guard_emergency_stop(self, ctx: BotContext) -> bool:
        # Only used as a manual override — always returns False in normal operation
        return False

    # ------------------------------------------------------------------
    # STATE MACHINE OPERATIONS
    # ------------------------------------------------------------------

    def get_valid_transitions(self, ctx: BotContext) -> list[Transition]:
        """Get all valid transitions from the current state given context."""
        valid = []
        for t in self.transitions:
            if t.from_state == self.state:
                if t.guard is None or t.guard(ctx):
                    valid.append(t)
        return valid

    def try_transition(self, ctx: BotContext) -> Optional[State]:
        """
        Attempt to transition to the next valid state.
        Returns the new state if transition occurred, None otherwise.
        Priority: NEWS_AVOID > IDLE (limits) > normal flow.
        """
        valid = self.get_valid_transitions(ctx)

        if not valid:
            return None

        # Priority ordering: NEWS_AVOID and IDLE (safety) take precedence
        priority_order = [State.NEWS_AVOID, State.IDLE, State.POSITION_ACTIVE,
                          State.ENTRY_SEARCH, State.COOLDOWN, State.SESSION_ANALYSIS,
                          State.TREND_MODE, State.RANGE_MODE]

        for priority_state in priority_order:
            for t in valid:
                if t.to_state == priority_state:
                    return self._execute_transition(t, ctx)

        # Fallback: take first valid
        return self._execute_transition(valid[0], ctx)

    def _execute_transition(self, transition: Transition, ctx: BotContext) -> State:
        """Execute a state transition and record history."""
        old_state = self.state
        self.state = transition.to_state
        self.history.append((ctx.current_time, old_state, self.state))
        return self.state

    def force_state(self, state: State, ctx: BotContext):
        """Force a state change (emergency use only)."""
        old_state = self.state
        self.state = state
        self.history.append((ctx.current_time, old_state, self.state))

    def get_state(self) -> State:
        return self.state

    def get_history(self, last_n: int = 10) -> list[tuple[datetime, State, State]]:
        return self.history[-last_n:]

    def reset(self):
        """Reset state machine to IDLE."""
        self.state = State.IDLE
        self.history.clear()

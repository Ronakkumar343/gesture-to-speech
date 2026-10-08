"""Unit tests for GestureStabilizer and DoublePalmStateMachine."""

import pytest
from stabilizer import DoublePalmStateMachine, GestureStabilizer


class MockClock:
    """Deterministic simulated clock for headless unit tests."""

    def __init__(self, start_time: float = 100.0) -> None:
        self.current_time = start_time

    def __call__(self) -> float:
        return self.current_time

    def advance(self, seconds: float) -> None:
        self.current_time += seconds


class TestGestureStabilizer:
    """Test suite for sliding-window majority voting and confidence filtering."""

    def test_steady_sequence_emits_winner(self):
        stabilizer = GestureStabilizer(window_size=5, min_confidence=0.6)
        # N=5, threshold=3
        out1 = stabilizer.update("Hello", 0.9)
        out2 = stabilizer.update("Hello", 0.9)
        assert out1 is None
        assert out2 is None

        # 3rd frame clears threshold (3 >= 3) and mean_conf (0.9 >= 0.6)
        out3 = stabilizer.update("Hello", 0.9)
        assert out3 == "Hello"

        # Subsequent frames continue emitting the stabilized gesture
        out4 = stabilizer.update("Hello", 0.9)
        assert out4 == "Hello"

    def test_low_confidence_does_not_emit(self):
        stabilizer = GestureStabilizer(window_size=5, min_confidence=0.7)
        # All frames have confidence 0.5 (below 0.7)
        for _ in range(5):
            res = stabilizer.update("Hello", 0.5)
            assert res is None
        assert stabilizer.last_winner is None

    def test_flicker_sequence_emits_nothing(self):
        """Alternating phrases between frames (flicker) must never emit."""
        stabilizer = GestureStabilizer(window_size=5, min_confidence=0.6)
        # Sequence of 10 alternating frames
        flicker_inputs = [("Hello", 0.9), ("Yes", 0.9)] * 5
        outputs = [stabilizer.update(phrase, conf) for phrase, conf in flicker_inputs]
        assert all(out is None for out in outputs)
        assert stabilizer.last_winner is None

    def test_tie_in_votes_emits_nothing(self):
        stabilizer = GestureStabilizer(window_size=4, vote_threshold=2, min_confidence=0.5)
        stabilizer.update("Hello", 0.9)
        stabilizer.update("Hello", 0.9)
        stabilizer.update("Yes", 0.9)
        res = stabilizer.update("Yes", 0.9)
        # Tie: Hello (2), Yes (2). No strict majority.
        assert res is None

    def test_hud_state_exposure(self):
        stabilizer = GestureStabilizer(window_size=5)
        for _ in range(3):
            stabilizer.update("Yes", 0.85)

        assert stabilizer.last_winner == "Yes"
        assert stabilizer.winner_votes == 3
        assert stabilizer.winner_confidence == 0.85
        assert len(stabilizer.window_contents) == 3

        state = stabilizer.state
        assert state["last_winner"] == "Yes"
        assert state["winner_votes"] == 3
        assert state["window_size"] == 5

    def test_clear_resets_stabilizer(self):
        stabilizer = GestureStabilizer(window_size=5)
        for _ in range(3):
            stabilizer.update("Stop", 0.9)
        assert stabilizer.last_winner == "Stop"

        stabilizer.clear()
        assert stabilizer.last_winner is None
        assert stabilizer.winner_votes == 0
        assert len(stabilizer.window_contents) == 0


class TestDoublePalmEmergencyHelp:
    """Test suite for temporal double-palm mapping to 'I need help'."""

    def test_single_palm_stays_hello(self):
        """Holding a single palm continuously across many frames must stay 'Hello'."""
        clock = MockClock(0.0)
        stabilizer = GestureStabilizer(window_size=5, clock=clock)

        results = []
        for _ in range(15):
            clock.advance(0.05)
            out = stabilizer.update("Hello", 0.95)
            if out:
                results.append(out)

        # Once threshold is reached, all emissions must remain 'Hello'
        assert len(results) >= 10
        assert all(r == "Hello" for r in results)
        assert "I need help" not in results

    def test_double_palm_within_window_emits_help(self):
        """Two palm presentations within 2.5s trigger 'I need help' on the second."""
        clock = MockClock(10.0)
        stabilizer = GestureStabilizer(window_size=5, double_palm_window=2.5, clock=clock)

        # 1st palm presentation (3 frames reaches threshold)
        r1 = None
        for _ in range(3):
            r1 = stabilizer.update("Hello", 0.9)
        assert r1 == "Hello"

        # Release hand (None frames)
        clock.advance(0.5)
        for _ in range(3):
            stabilizer.update(None, 0.0)

        # 2nd palm presentation at 10.0 + 0.5 = 10.5 (within 2.5s window)
        clock.advance(0.2)
        r2 = None
        for _ in range(3):
            r2 = stabilizer.update("Hello", 0.9)
        assert r2 == "I need help"

    def test_two_palms_too_far_apart_emit_two_hellos(self):
        """Two palms separated by more than 2.5s remain two separate 'Hello's."""
        clock = MockClock(0.0)
        stabilizer = GestureStabilizer(window_size=5, double_palm_window=2.5, clock=clock)

        # 1st palm
        for _ in range(3):
            stabilizer.update("Hello", 0.9)
        assert stabilizer.last_winner == "Hello"

        # Release hand and wait 3.5 seconds (> 2.5s)
        clock.advance(3.5)
        for _ in range(3):
            stabilizer.update(None, 0.0)

        # 2nd palm
        clock.advance(0.1)
        r2 = None
        for _ in range(3):
            r2 = stabilizer.update("Hello", 0.9)
        assert r2 == "Hello"

    def test_different_gesture_between_palms_resets_double_palm(self):
        """Palm followed by Fist then Palm does not trigger help."""
        clock = MockClock(0.0)
        stabilizer = GestureStabilizer(window_size=5, clock=clock)

        # 1st palm
        for _ in range(3):
            stabilizer.update("Hello", 0.9)
        assert stabilizer.last_winner == "Hello"

        # Intermediate gesture: Fist ("Stop")
        clock.advance(0.5)
        for _ in range(3):
            stabilizer.update("Stop", 0.9)
        assert stabilizer.last_winner == "Stop"

        # Release
        clock.advance(0.2)
        for _ in range(3):
            stabilizer.update(None, 0.0)

        # 2nd palm
        clock.advance(0.2)
        r = None
        for _ in range(3):
            r = stabilizer.update("Hello", 0.9)
        assert r == "Hello"


class TestDoublePalmStateMachineStandalone:
    """Test suite directly verifying the standalone DoublePalmStateMachine class."""

    def test_state_machine_transitions(self):
        clock = MockClock(0.0)
        sm = DoublePalmStateMachine(window_sec=2.0, clock=clock)

        # Initially None
        assert sm.update(None) is None

        # First palm onset
        assert sm.update("hello") == "Hello"
        # Holding palm
        assert sm.update("hello") == "Hello"

        # Release
        assert sm.update(None) is None

        # Advance within 2s window
        clock.advance(1.2)
        assert sm.update("hello") == "I need help"

        # Release
        assert sm.update(None) is None

        # 3rd palm immediately after help should be Hello (cycle restarted)
        clock.advance(0.2)
        assert sm.update("hello") == "Hello"

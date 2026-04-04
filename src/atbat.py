"""
atbat.py — Interactive at-bat session manager.

Tracks count, pitch sequence, and situational state across an at-bat.
Each pitch() call returns a hit probability from the trained model.

Usage (interactive CLI):
    python -m src.atbat

Usage (programmatic):
    from src.atbat import AtBat

    ab = AtBat(hitter="ohtani, shohei", p_throws="R", inning=7, score_diff=-1)
    result = ab.pitch(release_speed=95.2, release_spin_rate=2350,
                      pfx_x=-0.5, pfx_z=1.2,
                      release_pos_x=-1.8, release_pos_z=6.1,
                      release_extension=6.5,
                      plate_x=0.1, plate_z=2.8,
                      pitch_type="FF")
    print(result)
    ab.outcome("swinging_strike")

    result = ab.pitch(release_speed=88.0, ...)
"""

from src.model.predict import predict_hit_probability

# Valid outcome strings and their count effect
_BALL_OUTCOMES = {"ball", "blocked_ball", "pitchout"}
_STRIKE_OUTCOMES = {"called_strike", "swinging_strike", "swinging_strike_blocked", "foul_tip"}
# Foul balls add a strike only when < 2 strikes
_FOUL_OUTCOMES = {"foul", "foul_bunt", "foul_tip"}
# At-bat ending outcomes — stop accepting pitches
_ENDING_OUTCOMES = {
    "single", "double", "triple", "home_run",
    "field_out", "grounded_into_double_play", "double_play",
    "strikeout", "walk", "hit_by_pitch", "sac_fly", "sac_bunt",
    "force_out", "fielders_choice", "fielders_choice_out",
}


class AtBat:
    """
    Manages state across a single plate appearance.

    Parameters
    ----------
    hitter       : player name as it appears in Statcast (e.g. "ohtani, shohei")
    p_throws     : pitcher handedness — "R" or "L"
    hitter_stand : batter handedness — "R" or "L" (default "R")
    inning       : current inning number
    score_diff   : runs — batting team leads (positive) or trails (negative)
    on_1b/2b/3b  : True if runner on base, False otherwise
    """

    def __init__(
        self,
        hitter: str,
        p_throws: str = "R",
        hitter_stand: str = "R",
        inning: int = 1,
        score_diff: int = 0,
        on_1b: bool = False,
        on_2b: bool = False,
        on_3b: bool = False,
    ):
        self.hitter = hitter
        self.p_throws = p_throws
        self.hitter_stand = hitter_stand
        self.inning = inning
        self.score_diff = score_diff
        self.on_1b = on_1b
        self.on_2b = on_2b
        self.on_3b = on_3b

        self.balls = 0
        self.strikes = 0
        self.pitch_number = 0
        self.is_over = False

        self._prev_pitch_type = "FIRST_PITCH"
        self._prev_pitch_speed = -1.0
        self._prev_pitch_result = "FIRST_PITCH"

        self.history = []  # list of (pitch_dict, result_dict, outcome)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def pitch(
        self,
        release_speed: float,
        release_spin_rate: float,
        pfx_x: float,
        pfx_z: float,
        release_pos_x: float,
        release_pos_z: float,
        release_extension: float,
        plate_x: float,
        plate_z: float,
        pitch_type: str,
    ) -> dict:
        """
        Submit a pitch and get back hit probability.

        Returns dict with hit_probability, count, pitch_number, and profile metadata.
        Call outcome() afterwards to advance the count before the next pitch.
        """
        if self.is_over:
            raise RuntimeError("At-bat is over. Start a new AtBat.")

        self.pitch_number += 1

        pitch_dict = {
            "release_speed": release_speed,
            "release_spin_rate": release_spin_rate,
            "pfx_x": pfx_x,
            "pfx_z": pfx_z,
            "release_pos_x": release_pos_x,
            "release_pos_z": release_pos_z,
            "release_extension": release_extension,
            "plate_x": plate_x,
            "plate_z": plate_z,
            "pitch_type": pitch_type,
            "balls": self.balls,
            "strikes": self.strikes,
            "pitch_number": self.pitch_number,
            "prev_pitch_type": self._prev_pitch_type,
            "prev_pitch_speed": self._prev_pitch_speed,
            "prev_pitch_result": self._prev_pitch_result,
            "on_1b": self.on_1b or None,
            "on_2b": self.on_2b or None,
            "on_3b": self.on_3b or None,
            "inning": self.inning,
            "score_diff": self.score_diff,
            "p_throws": self.p_throws,
            "stand": self.hitter_stand,
        }

        result = predict_hit_probability(pitch_dict, self.hitter)
        result["pitch_number"] = self.pitch_number
        result["balls"] = self.balls
        result["strikes"] = self.strikes

        self.history.append({"pitch": pitch_dict, "prediction": result, "outcome": None})
        return result

    def outcome(self, result_str: str) -> None:
        """
        Record what happened on the last pitch and advance the count.

        result_str options:
          Balls   : "ball", "blocked_ball"
          Strikes : "called_strike", "swinging_strike", "swinging_strike_blocked"
          Fouls   : "foul", "foul_bunt"
          Ends PA : "single", "double", "triple", "home_run",
                    "strikeout", "walk", "field_out", etc.
        """
        if not self.history or self.history[-1]["outcome"] is not None:
            raise RuntimeError("Call pitch() before outcome().")

        self.history[-1]["outcome"] = result_str

        last_pitch = self.history[-1]["pitch"]
        self._prev_pitch_type = last_pitch["pitch_type"]
        self._prev_pitch_speed = last_pitch["release_speed"]
        self._prev_pitch_result = result_str

        if result_str in _ENDING_OUTCOMES:
            self.is_over = True
            return

        if result_str in _BALL_OUTCOMES:
            self.balls = min(self.balls + 1, 3)
            if self.balls == 4:  # walk — shouldn't happen but guard it
                self.is_over = True

        elif result_str in _STRIKE_OUTCOMES:
            self.strikes = min(self.strikes + 1, 2)
            if self.strikes == 3:
                self.is_over = True

        elif result_str in _FOUL_OUTCOMES:
            if self.strikes < 2:
                self.strikes += 1

    def summary(self) -> None:
        """Print a summary of all pitches thrown in this at-bat."""
        print(f"\n{'─'*55}")
        print(f"  {self.hitter.title()}  |  {self.p_throws}HP  |  "
              f"Inning {self.inning}  |  Score diff {self.score_diff:+d}")
        print(f"{'─'*55}")
        print(f"  {'#':<3} {'Type':<5} {'Velo':<6} {'Count':<6} {'Hit Prob':>8}  Outcome")
        print(f"{'─'*55}")
        for entry in self.history:
            p = entry["pitch"]
            pred = entry["prediction"]
            oc = entry["outcome"] or "—"
            count = f"{entry['prediction']['balls']}-{entry['prediction']['strikes']}"
            print(f"  {pred['pitch_number']:<3} {p['pitch_type']:<5} "
                  f"{p['release_speed']:<6.1f} {count:<6} "
                  f"{pred['hit_probability']:>7.1%}  {oc}")
        print(f"{'─'*55}\n")


# ---------------------------------------------------------------------------
# Interactive CLI
# ---------------------------------------------------------------------------

def _prompt(label: str, cast=str, default=None):
    suffix = f" [{default}]" if default is not None else ""
    raw = input(f"  {label}{suffix}: ").strip()
    if raw == "" and default is not None:
        return default
    return cast(raw)


def _setup_atbat() -> AtBat:
    print("\n=== NEW AT-BAT SETUP ===")
    hitter = _prompt("Hitter name (as in Statcast, e.g. 'ohtani, shohei')")
    p_throws = _prompt("Pitcher throws (R/L)", default="R").upper()
    hitter_stand = _prompt("Batter stands (R/L)", default="R").upper()
    inning = _prompt("Inning", cast=int, default=1)
    score_diff = _prompt("Score diff (batting team; + leads, - trails)", cast=int, default=0)
    on_1b = _prompt("Runner on 1B? (y/n)", default="n").lower() == "y"
    on_2b = _prompt("Runner on 2B? (y/n)", default="n").lower() == "y"
    on_3b = _prompt("Runner on 3B? (y/n)", default="n").lower() == "y"

    return AtBat(
        hitter=hitter,
        p_throws=p_throws,
        hitter_stand=hitter_stand,
        inning=inning,
        score_diff=score_diff,
        on_1b=on_1b,
        on_2b=on_2b,
        on_3b=on_3b,
    )


def _get_pitch() -> dict:
    print("\n--- PITCH INPUT ---")
    return {
        "pitch_type":          _prompt("Pitch type (FF/SI/SL/CU/CH/FC/FS/SV/KC)", default="FF").upper(),
        "release_speed":       _prompt("Velocity (mph)", cast=float),
        "release_spin_rate":   _prompt("Spin rate (rpm)", cast=float),
        "pfx_x":               _prompt("Horizontal movement pfx_x (ft, arm-side +)", cast=float, default=0.0),
        "pfx_z":               _prompt("Vertical movement pfx_z (ft, rise +)", cast=float, default=0.0),
        "release_pos_x":       _prompt("Release pos X (ft)", cast=float, default=0.0),
        "release_pos_z":       _prompt("Release pos Z (ft)", cast=float, default=6.0),
        "release_extension":   _prompt("Release extension (ft)", cast=float, default=6.0),
        "plate_x":             _prompt("Plate X location (ft, center=0)", cast=float, default=0.0),
        "plate_z":             _prompt("Plate Z location (ft, belt≈2.5)", cast=float, default=2.5),
    }


def _get_outcome(ab: AtBat) -> str:
    print("\n  Outcome options:")
    print("    ball / called_strike / swinging_strike / foul")
    print("    single / double / triple / home_run / field_out / strikeout / walk")
    return _prompt("Outcome").lower().replace(" ", "_")


def run_interactive():
    print("\n╔══════════════════════════════╗")
    print("║     PITCH DUEL SIMULATOR     ║")
    print("╚══════════════════════════════╝")

    while True:
        ab = _setup_atbat()
        print(f"\nAt-bat started: {ab.hitter.title()} vs {ab.p_throws}HP  |  "
              f"Count: {ab.balls}-{ab.strikes}\n")

        while not ab.is_over:
            pitch_data = _get_pitch()
            try:
                result = ab.pitch(**pitch_data)
            except Exception as e:
                print(f"\n  ERROR: {e}\n")
                continue

            print(f"\n  ┌─ HIT PROBABILITY: {result['hit_probability']:.1%} ──────────────────")
            print(f"  │  Hitter : {result['hitter'].title()}")
            print(f"  │  Pitch  : {result['pitch_type']}  |  Count: {result['count']}")
            if result.get("used_fallback"):
                print(f"  │  ⚠  League-average profile used (hitter not in database)")
            elif result.get("is_thin_sample"):
                print(f"  │  ⚠  Thin sample — profile blended with league averages")
            print(f"  └──────────────────────────────────────────────────────")

            oc = _get_outcome(ab)
            ab.outcome(oc)

            if not ab.is_over:
                print(f"\n  Count: {ab.balls}-{ab.strikes}  |  Pitch #{ab.pitch_number + 1}")

        ab.summary()

        again = input("New at-bat? (y/n): ").strip().lower()
        if again != "y":
            break

    print("\nSession ended.\n")


if __name__ == "__main__":
    run_interactive()

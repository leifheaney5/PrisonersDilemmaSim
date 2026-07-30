import json
import random
import unittest
from unittest.mock import patch

from pages.game_logic import (
    AdaptiveBestResponse,
    Appeaser,
    DebtCollector,
    EntropyBroker,
    EXTENDED_STRATEGY_NAMES,
    ForgetfulGrudger,
    Forgiver,
    Gradual,
    Handshake,
    HardMajority,
    HardTitForTat,
    PatternHunter,
    SoftMajority,
    TripleThreat,
    TwoTitsForTat,
    _init_strategy_state,
    payoff,
    init_human_match_state,
    init_tournament_state,
    list_strategy_names,
    make_strategy_factories,
    play_strategy,
    simulate_tournament,
    step_human_match,
    step_tournament,
)


class NewStrategyBehaviorTests(unittest.TestCase):
    def test_catalog_contains_one_hundred_unique_strategies(self):
        names = list_strategy_names()
        self.assertEqual(len(names), 100)
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(
            {
                "ForgetfulGrudger",
                "StochasticPavlov",
                "Appeaser",
                "Forgiver",
                "ReactivePlayer",
                "MemoryOnePlayer",
                "ZDExtort2",
                "ZDGenerous2",
                "ZDEqualizer",
                "Handshake",
                "AdaptiveBestResponse",
                "HedgeMetaStrategy",
            }
            <= set(names)
        )

    def test_all_forty_extended_strategies_can_complete_human_matches(self):
        human_moves = ["cooperate", "defect", "defect", "cooperate"] * 3
        self.assertEqual(len(EXTENDED_STRATEGY_NAMES), 40)
        for opponent in EXTENDED_STRATEGY_NAMES:
            with self.subTest(opponent=opponent):
                state = init_human_match_state(opponent=opponent, rounds=len(human_moves), seed=29)
                for move in human_moves:
                    state = step_human_match(state, human_move=move)
                self.assertTrue(state["done"])
                self.assertEqual(len(state["events"]), len(human_moves))
                self.assertTrue(
                    {event["opponent_move"] for event in state["events"]}
                    <= {"cooperate", "defect"}
                )

    def test_extended_profiles_include_explicit_philosophy(self):
        from pages.strategy_catalog import STRATEGY_PROFILES

        for name in EXTENDED_STRATEGY_NAMES:
            with self.subTest(strategy=name):
                self.assertTrue(STRATEGY_PROFILES[name]["description"])
                self.assertTrue(STRATEGY_PROFILES[name]["philosophy"])

    def test_extended_periodic_and_number_schedules_have_documented_signatures(self):
        factory_by_name = {factory().name: factory for factory in make_strategy_factories()}
        expected = {
            "PeriodicCCD": "CCDCCDCC",
            "PeriodicDDC": "DDCDDCDD",
            "PeriodicCCCD": "CCCDCCCD",
            "PeriodicCDDD": "CDDDCDDD",
            "PrimeCooperator": "DCCDCDCD",
            "SquareDefector": "DCCDCCCC",
        }
        for name, signature in expected.items():
            strategy = factory_by_name[name]()
            history = []
            moves = []
            for _ in signature:
                moves.append("C" if strategy.play(history) == "cooperate" else "D")
                history.append("cooperate")
            with self.subTest(strategy=name):
                self.assertEqual("".join(moves), signature)

    def test_three_tits_for_tat_does_not_extend_active_punishment(self):
        factory_by_name = {factory().name: factory for factory in make_strategy_factories()}
        strategy = factory_by_name["ThreeTitsForTat"]()
        history = []
        sequence = []
        for observed in ("defect", "defect", "cooperate", "cooperate"):
            sequence.append(strategy.play(history))
            history.append(observed)
        sequence.append(strategy.play(history))
        self.assertEqual(sequence, ["cooperate", "defect", "defect", "defect", "cooperate"])

    def test_three_strikes_punishes_each_third_offense_once(self):
        factory_by_name = {factory().name: factory for factory in make_strategy_factories()}
        strategy = factory_by_name["ThreeStrikes"]()
        history = []
        outputs = []
        for observed in ("defect", "defect", "defect", "cooperate", "defect"):
            outputs.append(strategy.play(history))
            history.append(observed)
        outputs.append(strategy.play(history))
        self.assertEqual(outputs, ["cooperate", "cooperate", "cooperate", "defect", "cooperate", "cooperate"])

    def test_shootout_and_bad_alternator_use_opposite_phases(self):
        factory_by_name = {factory().name: factory for factory in make_strategy_factories()}
        signatures = {}
        for name in ("BadAlternator", "Shootout"):
            strategy = factory_by_name[name]()
            signatures[name] = [strategy.play([]) for _ in range(4)]
        self.assertEqual(signatures["BadAlternator"], ["cooperate", "defect", "cooperate", "defect"])
        self.assertEqual(signatures["Shootout"], ["defect", "cooperate", "defect", "cooperate"])

    def test_all_twelve_new_strategies_can_play_a_human_match(self):
        for opponent in (
            "ForgetfulGrudger",
            "StochasticPavlov",
            "Appeaser",
            "Forgiver",
            "ReactivePlayer",
            "MemoryOnePlayer",
            "ZDExtort2",
            "ZDGenerous2",
            "ZDEqualizer",
            "Handshake",
            "AdaptiveBestResponse",
            "HedgeMetaStrategy",
        ):
            with self.subTest(opponent=opponent):
                state = init_human_match_state(opponent=opponent, rounds=2, seed=9)
                state = step_human_match(state, human_move="cooperate")
                self.assertIn(state["opponent_history"][-1], {"cooperate", "defect"})

    def test_forgetful_grudger_returns_to_cooperation(self):
        strategy = ForgetfulGrudger("ForgetfulGrudger", punishment_length=3)
        histories = [[], ["defect"], ["defect", "cooperate"], ["defect", "cooperate", "cooperate"], ["defect", "cooperate", "cooperate", "cooperate"]]
        self.assertEqual(
            [strategy.play(history) for history in histories],
            ["cooperate", "defect", "defect", "defect", "cooperate"],
        )

    def test_appeaser_switches_after_defection(self):
        strategy = Appeaser("Appeaser")
        self.assertEqual(strategy.play([]), "cooperate")
        self.assertEqual(strategy.play(["defect"]), "defect")
        self.assertEqual(strategy.play(["defect", "cooperate"]), "defect")
        self.assertEqual(strategy.play(["defect", "cooperate", "defect"]), "cooperate")

    def test_forgiver_punishes_once_then_reconciles(self):
        strategy = Forgiver("Forgiver")
        histories = [[], ["defect"], ["defect", "defect"], ["defect", "defect", "cooperate"]]
        self.assertEqual(
            [strategy.play(history) for history in histories],
            ["cooperate", "defect", "cooperate", "cooperate"],
        )

    def test_handshake_recognizes_matching_opening(self):
        matching = Handshake("Handshake")
        history = []
        moves = []
        for opponent_move in ["cooperate", "defect", "cooperate", "cooperate", "cooperate"]:
            moves.append(matching.play(history))
            history.append(opponent_move)
        self.assertEqual(moves, ["cooperate", "defect", "cooperate", "cooperate", "cooperate"])
        self.assertEqual(Handshake("Handshake").play(["cooperate", "cooperate", "cooperate", "cooperate"]), "defect")

    def test_adaptive_best_response_updates_conditional_counts(self):
        strategy = AdaptiveBestResponse("AdaptiveBestResponse")
        first = strategy.play([])
        second = strategy.play(["cooperate"])
        self.assertIn(first, {"cooperate", "defect"})
        self.assertIn(second, {"cooperate", "defect"})
        self.assertEqual(strategy.seen, 1)

    def test_debt_collector_uses_repayable_trust_ledger(self):
        strategy = DebtCollector("DebtCollector")
        histories = [[], ["defect"], ["defect", "cooperate"], ["defect", "cooperate", "cooperate"]]
        self.assertEqual([strategy.play(history) for history in histories], ["cooperate", "defect", "cooperate", "cooperate"])

    def test_pattern_hunter_learns_transition_after_current_move(self):
        strategy = PatternHunter("PatternHunter")
        self.assertEqual(strategy.play([]), "cooperate")
        self.assertEqual(strategy.play(["cooperate", "defect", "cooperate", "defect", "cooperate"]), "defect")

    def test_entropy_broker_shields_against_rapid_switching(self):
        strategy = EntropyBroker("EntropyBroker")
        self.assertEqual(strategy.play(["cooperate", "defect", "cooperate", "defect"]), "defect")
        self.assertEqual(strategy.play(["cooperate", "cooperate", "cooperate", "defect"]), "cooperate")

    def test_custom_policy_composes_response_retaliation_and_threshold(self):
        state = _init_strategy_state(
            "LedgerLab",
            {
                "start_move": "cooperate",
                "response_mode": "anti_tft",
                "use_tft": False,
                "use_grudge": False,
                "retaliation_window": 2,
                "threshold_enabled": False,
                "defect_rate_threshold": 0.5,
                "min_history": 3,
                "endgame_after_turn": 0,
                "noise": 0.0,
            },
        )
        move, _, _ = play_strategy("LedgerLab", ["defect", "cooperate"], state, 1)
        self.assertEqual(move, "defect")

    def test_two_tits_for_tat_retaliates_twice(self):
        strategy = TwoTitsForTat("TwoTitsForTat")
        histories = [
            [],
            ["cooperate"],
            ["cooperate", "defect"],
            ["cooperate", "defect", "cooperate"],
            ["cooperate", "defect", "cooperate", "cooperate"],
        ]
        self.assertEqual(
            [strategy.play(history) for history in histories],
            ["cooperate", "cooperate", "defect", "defect", "cooperate"],
        )

    def test_hard_tit_for_tat_remembers_three_rounds(self):
        strategy = HardTitForTat("HardTitForTat")
        histories = [
            [],
            ["defect"],
            ["defect", "cooperate"],
            ["defect", "cooperate", "cooperate"],
            ["defect", "cooperate", "cooperate", "cooperate"],
        ]
        self.assertEqual(
            [strategy.play(history) for history in histories],
            ["cooperate", "defect", "defect", "defect", "cooperate"],
        )

    def test_majority_variants_have_opposite_tie_breakers(self):
        soft = SoftMajority("SoftMajority")
        hard = HardMajority("HardMajority")
        tied = ["cooperate", "defect"]
        cooperative_majority = ["cooperate", "cooperate", "defect"]
        defective_majority = ["cooperate", "defect", "defect"]

        self.assertEqual(soft.play([]), "cooperate")
        self.assertEqual(hard.play([]), "defect")
        self.assertEqual(soft.play(tied), "cooperate")
        self.assertEqual(hard.play(tied), "defect")
        self.assertEqual(soft.play(cooperative_majority), "cooperate")
        self.assertEqual(hard.play(cooperative_majority), "cooperate")
        self.assertEqual(soft.play(defective_majority), "defect")
        self.assertEqual(hard.play(defective_majority), "defect")

    def test_gradual_increases_punishment_then_calms(self):
        strategy = Gradual("Gradual")
        opponent_moves = [
            "defect",
            "cooperate",
            "cooperate",
            "defect",
            "cooperate",
            "cooperate",
            "cooperate",
            "cooperate",
        ]
        history = []
        actual = [strategy.play(history)]
        for move in opponent_moves:
            history = [*history, move]
            actual.append(strategy.play(history))

        self.assertEqual(
            actual,
            [
                "cooperate",
                "defect",
                "cooperate",
                "cooperate",
                "defect",
                "defect",
                "cooperate",
                "cooperate",
                "cooperate",
            ],
        )

    def test_gradual_extends_outstanding_punishment(self):
        strategy = Gradual("Gradual")
        history = []
        actual = []
        for opponent_move in ["defect", "defect", "cooperate", "cooperate", "cooperate", "cooperate"]:
            actual.append(strategy.play(history))
            history.append(opponent_move)
        actual.append(strategy.play(history))

        # The first offense adds one punishment round. The consecutive second
        # offense adds two more, followed by two calming cooperation rounds.
        self.assertEqual(
            actual,
            ["cooperate", "defect", "defect", "defect", "cooperate", "cooperate", "cooperate"],
        )

    def test_new_strategies_are_registered(self):
        names = set(list_strategy_names())
        expected = {"Gradual", "SoftMajority", "HardMajority", "HardTitForTat", "TwoTitsForTat"}
        self.assertTrue(expected <= names)


class IncrementalParityTests(unittest.TestCase):
    def assert_policy_parity(self, strategy, opponent_moves):
        state = _init_strategy_state(strategy.name)
        rng_state = 7
        history = []
        for opponent_move in opponent_moves:
            class_move = strategy.play(history)
            incremental_move, state, rng_state = play_strategy(
                strategy.name, history, state, rng_state
            )
            self.assertEqual(
                class_move,
                incremental_move,
                f"{strategy.name} diverged after history {history}",
            )
            history = [*history, opponent_move]

        class_move = strategy.play(history)
        incremental_move, _, _ = play_strategy(strategy.name, history, state, rng_state)
        self.assertEqual(
            class_move,
            incremental_move,
            f"{strategy.name} diverged after history {history}",
        )

    def test_new_strategy_batch_and_incremental_policies_match(self):
        opponent_moves = [
            "cooperate",
            "defect",
            "cooperate",
            "defect",
            "defect",
            "cooperate",
            "cooperate",
            "cooperate",
        ]
        for strategy in (
            Gradual("Gradual"),
            SoftMajority("SoftMajority"),
            HardMajority("HardMajority"),
            HardTitForTat("HardTitForTat"),
            TwoTitsForTat("TwoTitsForTat"),
        ):
            with self.subTest(strategy=strategy.name):
                self.assert_policy_parity(strategy, opponent_moves)

    def test_triple_threat_batch_and_incremental_policies_match(self):
        self.assert_policy_parity(TripleThreat("TripleThreat"), ["cooperate"] * 11)

    def test_every_deterministic_strategy_has_batch_incremental_parity(self):
        from pages.strategy_catalog import strategy_scorecard

        opponent_moves = [
            "cooperate",
            "defect",
            "defect",
            "cooperate",
            "cooperate",
            "defect",
            "cooperate",
            "defect",
            "cooperate",
            "cooperate",
            "cooperate",
            "defect",
        ]
        for factory in make_strategy_factories():
            strategy = factory()
            if not strategy_scorecard(strategy.name)["deterministic"]:
                continue
            with self.subTest(strategy=strategy.name):
                self.assert_policy_parity(strategy, opponent_moves)


class CatalogIntegrationTests(unittest.TestCase):
    def test_profiles_and_scorecards_cover_every_registered_strategy(self):
        from pages.strategy_catalog import STRATEGY_PROFILES, strategy_scorecard

        names = set(list_strategy_names())
        self.assertEqual(names, set(STRATEGY_PROFILES))
        for name in names:
            scorecard = strategy_scorecard(name)
            with self.subTest(strategy=name):
                self.assertNotEqual(scorecard["memory"], "unknown")
                self.assertNotEqual(scorecard["deterministic"], scorecard["stochastic"])

    def test_catalog_canonicalizes_legacy_aliases(self):
        from pages.strategy_catalog import STRATEGY_ALIASES, canonical_strategy_name

        for alias, canonical_name in STRATEGY_ALIASES.items():
            with self.subTest(alias=alias):
                self.assertEqual(canonical_strategy_name(alias), canonical_name)
                self.assertIn(canonical_name, list_strategy_names())

    def test_registered_names_and_aliases_are_unique(self):
        from pages.strategy_catalog import STRATEGY_ALIASES

        names = list_strategy_names()
        self.assertEqual(len(names), len(set(names)))
        self.assertFalse(set(STRATEGY_ALIASES) & set(names))

    def test_catalog_identifies_horizon_aware_strategies(self):
        from pages.strategy_catalog import HORIZON_AWARE_STRATEGIES, strategy_scorecard

        names = set(list_strategy_names())
        self.assertLessEqual(HORIZON_AWARE_STRATEGIES, names)
        for name in names:
            with self.subTest(strategy=name):
                self.assertEqual(
                    strategy_scorecard(name)["horizon_aware"],
                    name in HORIZON_AWARE_STRATEGIES,
                )

    def test_batch_tournament_includes_all_registered_pairings(self):
        names = list_strategy_names()
        result = simulate_tournament(rounds_per_match=2, repetitions=1, seed=11)
        expected_pairings = len(names) * (len(names) - 1) // 2

        self.assertEqual(len(result), expected_pairings * 2)
        self.assertEqual(set(result["strategy_1"]) | set(result["strategy_2"]), set(names))
        self.assertTrue(set(result["move_1"]) <= {"cooperate", "defect"})
        self.assertTrue(set(result["move_2"]) <= {"cooperate", "defect"})

    def test_incremental_tournament_completes_with_new_strategies(self):
        names = ["Gradual", "SoftMajority", "HardMajority", "HardTitForTat", "TwoTitsForTat"]
        state = init_tournament_state(
            strategy_names=names,
            rounds_per_match=8,
            repetitions=2,
            seed=13,
        )
        while not state["done"]:
            state = step_tournament(state, max_rounds=7)

        expected_matches = 2 * (len(names) * (len(names) - 1) // 2)
        self.assertEqual(state["matches_done"], expected_matches)
        self.assertEqual(sum(state["rounds_played"].values()), expected_matches * 8 * 2)
        self.assertEqual(set(state["totals"]), set(names))

    def test_human_matches_support_every_new_strategy(self):
        names = ["Gradual", "SoftMajority", "HardMajority", "HardTitForTat", "TwoTitsForTat"]
        human_moves = ["cooperate", "defect", "cooperate", "defect", "cooperate"]
        for name in names:
            with self.subTest(strategy=name):
                state = init_human_match_state(opponent=name, rounds=len(human_moves), seed=17)
                for move in human_moves:
                    state = step_human_match(state, human_move=move)
                self.assertTrue(state["done"])
                self.assertEqual(state["round"], len(human_moves))
                self.assertEqual(len(state["events"]), len(human_moves))
                self.assertTrue(
                    {event["opponent_move"] for event in state["events"]}
                    <= {"cooperate", "defect"}
                )

    def test_horizon_metadata_is_only_supplied_when_known(self):
        from pages.strategy_catalog import HORIZON_AWARE_STRATEGIES

        for name in HORIZON_AWARE_STRATEGIES:
            with self.subTest(strategy=name):
                known = init_human_match_state(opponent=name, rounds=4, seed=19, horizon_known=True)
                unknown = init_human_match_state(opponent=name, rounds=4, seed=19, horizon_known=False)
                self.assertEqual(known["opponent_state"]["match_total_rounds"], 4)
                self.assertIsNone(unknown["opponent_state"]["match_total_rounds"])


class ValidationTests(unittest.TestCase):
    def test_execution_error_rate_is_bounded(self):
        for value in (-0.01, 1.01, float("nan")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "execution_error_rate"):
                    init_tournament_state(
                        strategy_names=["MrNiceGuy", "BadCop"],
                        rounds_per_match=1,
                        repetitions=1,
                        execution_error_rate=value,
                    )
                with self.assertRaisesRegex(ValueError, "execution_error_rate"):
                    init_human_match_state(opponent="MrNiceGuy", execution_error_rate=value)

    def test_custom_strategy_configuration_is_normalized(self):
        state = init_human_match_state(
            opponent="Custom",
            custom_strategies={
                "Custom": {
                    "start_move": "cooperate",
                    "use_tft": True,
                    "min_history": "3",
                    "defect_rate_threshold": "0.25",
                }
            },
        )
        self.assertEqual(
            state["custom_strategies"]["Custom"],
            {
                "start_move": "cooperate",
                "use_tft": True,
                "use_grudge": False,
                "response_mode": "tft",
                "retaliation_window": 0,
                "threshold_enabled": True,
                "defect_rate_threshold": 0.25,
                "min_history": 3,
                "endgame_after_turn": 0,
                "noise": 0.0,
            },
        )

    def test_custom_strategy_rejects_invalid_fields_and_values(self):
        cases = (
            ({"start_move": "maybe"}, "start_move"),
            ({"noise": float("nan")}, "between 0 and 1"),
            ({"noise": float("inf")}, "between 0 and 1"),
            ({"use_tft": "yes"}, "must be a boolean"),
            ({"min_history": True}, "must be an integer"),
            ({"noise": True}, "must be a number"),
            ({"surprise": 1}, "unknown field"),
        )
        for config, message in cases:
            with self.subTest(config=config):
                with self.assertRaisesRegex(ValueError, message):
                    init_human_match_state(
                        opponent="Custom",
                        custom_strategies={"Custom": config},
                    )

    def test_incremental_limits_are_validated(self):
        base = dict(
            strategy_names=["TitForTat", "BadCop"],
            rounds_per_match=5,
            repetitions=1,
        )
        for field, value in (("recent_limit", -1), ("timeline_limit", -1), ("timeline_stride", 0)):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, field):
                    init_tournament_state(**base, **{field: value})

        with self.assertRaisesRegex(ValueError, "at most 10"):
            init_tournament_state(
                strategy_names=list_strategy_names()[:11],
                rounds_per_match=5,
                repetitions=1,
            )

        with self.assertRaisesRegex(ValueError, "workload"):
            init_tournament_state(
                strategy_names=list_strategy_names()[:10],
                rounds_per_match=1_000,
                repetitions=1_000,
            )

    def test_custom_strategy_name_must_not_be_whitespace(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            init_human_match_state(opponent=" ", custom_strategies={" ": {}})

    def test_duplicate_strategy_selection_is_rejected_after_alias_normalization(self):
        with self.assertRaisesRegex(ValueError, "Duplicate strategy"):
            init_tournament_state(
                strategy_names=["Pushover", "ThePushover"],
                rounds_per_match=5,
                repetitions=1,
            )

    def test_custom_strategy_cannot_shadow_a_builtin(self):
        with self.assertRaisesRegex(ValueError, "conflicts with a built-in"):
            init_human_match_state(
                opponent="TitForTat",
                custom_strategies={"TitForTat": {"start_move": "defect"}},
            )

    def test_custom_strategy_configuration_must_be_a_dictionary(self):
        with self.assertRaisesRegex(ValueError, "configuration must be a dictionary"):
            init_human_match_state(
                opponent="MyStrategy",
                custom_strategies={"MyStrategy": "invalid"},  # type: ignore[dict-item]
            )

    def test_custom_strategy_numeric_settings_are_bounded(self):
        for field, value in (("noise", 1.1), ("defect_rate_threshold", -0.1)):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "between 0 and 1"):
                    init_human_match_state(
                        opponent="MyStrategy",
                        custom_strategies={"MyStrategy": {field: value}},
                    )

    def test_unknown_incremental_strategy_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown strategy"):
            play_strategy("TypoForTat", [], {}, 0)

    def test_unknown_tournament_strategy_is_rejected_during_initialization(self):
        with self.assertRaisesRegex(ValueError, "TypoForTat"):
            init_tournament_state(
                strategy_names=["TitForTat", "TypoForTat"],
                rounds_per_match=5,
                repetitions=1,
            )

    def test_unknown_human_opponent_is_rejected_during_initialization(self):
        with self.assertRaisesRegex(ValueError, "TypoForTat"):
            init_human_match_state(opponent="TypoForTat")

    def test_custom_strategy_names_remain_valid(self):
        config = {"MyStrategy": {"start_move": "cooperate"}}
        state = init_human_match_state(opponent="MyStrategy", custom_strategies=config)
        state = step_human_match(state, human_move="cooperate")
        self.assertEqual(state["opponent_history"], ["cooperate"])

    def test_payoff_rejects_invalid_moves(self):
        with self.assertRaisesRegex(ValueError, "Player 1 returned invalid move"):
            payoff("invalid", "cooperate")  # type: ignore[arg-type]

    def test_human_match_rejects_invalid_moves(self):
        state = init_human_match_state(opponent="TitForTat")
        with self.assertRaisesRegex(ValueError, "Human player returned invalid move"):
            step_human_match(state, human_move="invalid")  # type: ignore[arg-type]

    def test_horizon_configuration_errors_are_not_silenced(self):
        class BrokenHorizonStrategy:
            name = "BrokenHorizon"

            def set_total_rounds(self, total_rounds):
                raise RuntimeError("broken horizon")

            def play(self, opponent_history):
                return "cooperate"

        with patch(
            "pages.game_logic.make_strategy_factories",
            return_value=[
                lambda: BrokenHorizonStrategy(),
                lambda: BrokenHorizonStrategy(),
            ],
        ):
            with self.assertRaisesRegex(RuntimeError, "broken horizon"):
                simulate_tournament(rounds_per_match=2, repetitions=1)


class ReproducibilityTests(unittest.TestCase):
    def test_batch_tournament_can_limit_registered_strategies(self):
        results = simulate_tournament(
            strategy_names=["TitForTat", "BadCop", "MrNiceGuy"],
            rounds_per_match=1,
            repetitions=1,
            include_self_play=True,
        )
        names = set(results["strategy_1"]) | set(results["strategy_2"])
        self.assertEqual(names, {"TitForTat", "BadCop", "MrNiceGuy"})
        self.assertEqual(len(results), 6)

    def test_batch_tournament_validates_selected_strategies(self):
        with self.assertRaisesRegex(ValueError, "duplicates"):
            simulate_tournament(strategy_names=["TitForTat", "TitForTat"])
        with self.assertRaisesRegex(ValueError, "Unknown"):
            simulate_tournament(strategy_names=["TitForTat", "Missing"])
        with self.assertRaisesRegex(ValueError, "batch tournament workload"):
            simulate_tournament(
                strategy_names=["TitForTat", "BadCop", "MrNiceGuy"],
                rounds_per_match=1_000,
                repetitions=1_000,
            )

    def test_execution_error_records_intended_and_executed_moves(self):
        results = simulate_tournament(rounds_per_match=1, repetitions=1, seed=7, execution_error_rate=1.0)
        self.assertTrue((results["intended_move_1"] != results["move_1"]).all())
        self.assertTrue((results["intended_move_2"] != results["move_2"]).all())

    def test_execution_error_is_reproducible(self):
        first = simulate_tournament(rounds_per_match=2, repetitions=1, seed=11, execution_error_rate=0.25)
        second = simulate_tournament(rounds_per_match=2, repetitions=1, seed=11, execution_error_rate=0.25)
        self.assertTrue(first.equals(second))

    def test_batch_execution_error_updates_strategy_self_memory(self):
        results = simulate_tournament(
            strategy_names=["WinStayLoseShift", "MrNiceGuy"],
            rounds_per_match=2,
            repetitions=1,
            execution_error_rate=1.0,
        )
        self.assertEqual(results["intended_move_1"].tolist(), ["cooperate", "cooperate"])

    def test_human_execution_error_preserves_intention(self):
        state = init_human_match_state(opponent="MrNiceGuy", rounds=1, seed=3, execution_error_rate=1.0)
        finished = step_human_match(state, human_move="cooperate")
        event = finished["events"][0]
        self.assertEqual(event["intended_human_move"], "cooperate")
        self.assertEqual(event["human_move"], "defect")
        self.assertEqual(event["intended_opponent_move"], "cooperate")
        self.assertEqual(event["opponent_move"], "defect")

    def test_human_execution_error_updates_opponent_self_memory(self):
        state = init_human_match_state(
            opponent="WinStayLoseShift", rounds=2, seed=3, execution_error_rate=1.0
        )
        state = step_human_match(state, human_move="cooperate")
        state = step_human_match(state, human_move="cooperate")
        self.assertEqual(
            [event["intended_opponent_move"] for event in state["events"]],
            ["cooperate", "cooperate"],
        )

    def test_self_play_adds_diagonal_pairings(self):
        names = list_strategy_names()
        results = simulate_tournament(rounds_per_match=1, repetitions=1, include_self_play=True)
        pairs = set(zip(results["strategy_1"], results["strategy_2"]))
        self.assertEqual(len(pairs), len(names) * (len(names) + 1) // 2)
        self.assertTrue(all((name, name) in pairs for name in names))

    def test_incremental_self_play_uses_expected_match_count(self):
        state = init_tournament_state(
            strategy_names=["MrNiceGuy", "BadCop"],
            rounds_per_match=1,
            repetitions=1,
            include_self_play=True,
        )
        self.assertEqual(state["total_matches"], 3)
        while not state["done"]:
            state = step_tournament(state, max_rounds=1)
        self.assertEqual(state["matches_done"], 3)

    def test_incremental_error_records_intended_and_executed_moves(self):
        state = init_tournament_state(
            strategy_names=["MrNiceGuy", "BadCop"],
            rounds_per_match=1,
            repetitions=1,
            execution_error_rate=1.0,
        )
        state = step_tournament(state, max_rounds=1)
        self.assertEqual(state["recent_format"], "compact_v2")
        row = state["recent"][0]
        self.assertEqual(row[4:8], [0, 1, 1, 0])

    def test_incremental_error_updates_strategy_self_memory(self):
        state = init_tournament_state(
            strategy_names=["WinStayLoseShift", "MrNiceGuy"],
            rounds_per_match=2,
            repetitions=1,
            execution_error_rate=1.0,
        )
        state = step_tournament(state, max_rounds=2)
        self.assertEqual([row[4] for row in state["recent"]], [0, 0])

    @staticmethod
    def _complete_tournament(max_rounds):
        state = init_tournament_state(
            strategy_names=["ImSoRandom", "MarkedMan", "RandomStranger", "Lottery"],
            rounds_per_match=9,
            repetitions=2,
            seed=2026,
        )
        while not state["done"]:
            state = step_tournament(state, max_rounds=max_rounds)
        return state

    def test_batch_tournament_is_reproducible_with_the_same_seed(self):
        first = simulate_tournament(rounds_per_match=3, repetitions=1, seed=101)
        second = simulate_tournament(rounds_per_match=3, repetitions=1, seed=101)
        self.assertTrue(first.equals(second))

    def test_batch_tournament_changes_with_a_different_seed(self):
        first = simulate_tournament(rounds_per_match=4, repetitions=1, seed=101)
        second = simulate_tournament(rounds_per_match=4, repetitions=1, seed=102)
        self.assertFalse(first.equals(second))

    def test_batch_tournament_preserves_the_callers_random_state(self):
        random.seed(505)
        before = random.getstate()
        simulate_tournament(rounds_per_match=2, repetitions=1, seed=606)
        self.assertEqual(random.getstate(), before)

    def test_incremental_result_does_not_depend_on_chunk_size(self):
        self.assertEqual(self._complete_tournament(1), self._complete_tournament(37))

    def test_incremental_tournament_resumes_after_json_round_trip(self):
        state = init_tournament_state(
            strategy_names=["ImSoRandom", "MarkedMan", "Lottery"],
            rounds_per_match=7,
            repetitions=2,
            seed=303,
        )
        state = step_tournament(state, max_rounds=11)
        restored = json.loads(json.dumps(state))

        uninterrupted = step_tournament(state, max_rounds=10_000)
        resumed = step_tournament(restored, max_rounds=10_000)
        self.assertEqual(uninterrupted, resumed)

    def test_human_match_resumes_after_json_round_trip(self):
        moves = ["cooperate", "defect", "cooperate", "cooperate", "defect"]
        state = init_human_match_state(opponent="ImSoRandom", rounds=len(moves), seed=404)
        state = step_human_match(state, human_move=moves[0])
        restored = json.loads(json.dumps(state))

        for move in moves[1:]:
            state = step_human_match(state, human_move=move)
            restored = step_human_match(restored, human_move=move)
        self.assertEqual(state, restored)


if __name__ == "__main__":
    unittest.main()

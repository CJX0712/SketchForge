"""Tests for the closed-form power-law error-law fit."""

from __future__ import annotations

import numpy as np

from sketchforge.training.fit_error_model import fit_error_law_closed_form


def test_recovers_known_exponent():
    budgets = np.array([64, 128, 256, 512, 1024, 2048], dtype=float)
    a, c = 3.0, 0.7
    errors = a * np.power(budgets, -c)
    law = fit_error_law_closed_form(budgets, errors)
    assert not law.degenerate
    assert law.c > 0  # err = a*b^{-c}: c > 0 means error falls with budget
    assert abs(law.c - c) < 1e-6
    assert law.r2 > 0.999


def test_predict_matches_synthetic_curve():
    budgets = np.array([64, 128, 256, 512, 1024], dtype=float)
    a, c = 2.5, 0.5
    errors = a * np.power(budgets, -c)
    law = fit_error_law_closed_form(budgets, errors)
    assert abs(law.predict(256) - 2.5 * 256 ** -0.5) < 1e-6
    # never below floor, and finite at budget 0
    assert law.predict(0) > 0
    assert np.isfinite(law.predict(10**9))


def test_degenerate_on_increasing_error():
    budgets = np.array([64, 128, 256, 512], dtype=float)
    errors = 0.1 * np.power(budgets, 0.5)  # error grows with budget
    law = fit_error_law_closed_form(budgets, errors)
    assert law.c <= 0
    assert law.degenerate


def test_degenerate_on_low_r2():
    rng = np.random.default_rng(0)
    budgets = np.array([64, 128, 256, 512, 1024], dtype=float)
    errors = rng.random(5)  # pure noise
    law = fit_error_law_closed_form(budgets, errors)
    assert law.degenerate  # r2 < 0.9


def test_elasticity_sign_convention():
    # A well-fit decreasing-error law exposes a positive elasticity magnitude.
    budgets = np.array([64, 128, 256, 512, 1024], dtype=float)
    law = fit_error_law_closed_form(budgets, 1.0 * np.power(budgets, -0.4))
    assert law.elasticity() > 0

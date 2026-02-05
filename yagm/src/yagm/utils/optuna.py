import gc
import math
import multiprocessing

import optuna
from IPython.core.display_functions import display
from joblib import Parallel, delayed
from optuna.trial import Trial, TrialState
from tqdm import tqdm
import logging
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)


def optuna_check_duplicate(trial: Trial):
    # Fetch all the trials to consider.
    # In this example, we use only completed trials, but users can specify other states
    # such as TrialState.PRUNED and TrialState.FAIL.
    states_to_consider = (TrialState.COMPLETE,)
    trials_to_consider = trial.study.get_trials(
        deepcopy=False, states=states_to_consider
    )
    # Check whether we already evaluated the sampled `(x, y)`.
    for t in reversed(trials_to_consider):
        if trial.params == t.params:
            # Use the existing value as trial duplicated the parameters.
            return t.values
    return None


def summary_study(
    study: optuna.study.Study,
    metrics: Tuple[str],
    directions: Tuple[str],
    top=3,
    prune_duplicate=True,
):
    assert len(metrics) == len(directions)
    assert all([e in ["minimize", "maximize"] for e in directions])
    logger.info("Number of finished trials: %d", len(study.trials))
    logger.info(
        "Number of best trials (located at the Pareto front): %d",
        len(study.best_trials),
    )
    if prune_duplicate:
        added = []
        best_trials = []
        for trial in tqdm(study.best_trials, desc="Prune duplicate"):
            if (trial.params, trial.values) in added:
                continue
            else:
                added.append((trial.params, trial.values))
                best_trials.append(trial)
        del added
        gc.collect()
    else:
        best_trials = study.best_trials

    best_trials = sorted(best_trials, key=lambda t: t.values[0], reverse=True)[:top]
    for i, trial in enumerate(best_trials):
        logger.info(
            "==========TOP %d==========\n\t***Trial Number: %d\n\t***Values=%s\n\t***Params: %s\n=====================",
            i,
            trial.number,
            trial.values,
            trial.params,
        )
    try:
        display(
            optuna.visualization.plot_pareto_front(
                study, target_names=study.metric_names
            )
        )
    except Exception as e:
        logger.warning("Exception occur while plotting Pareto Front: %s", e)
    for metric_idx, metric_name in enumerate(metrics):
        logger.info("metric=%s index=%d", metric_name, metric_idx)
        try:
            display(
                optuna.visualization.plot_param_importances(
                    study,
                    target=lambda t: t.values[metric_idx],
                    target_name=metric_name,
                )
            )
        except Exception as e:
            logger.warning("Exception occur while plotting Params Importances: %s", e)

    df = (
        study.trials_dataframe(
            attrs=(
                "number",
                "value",
                "datetime_start",
                "datetime_complete",
                "duration",
                "params",
                "user_attrs",
                "system_attrs",
                "state",
            ),
            multi_index=False,
        )
        .sort_values(
            by=metrics,
            # ascending=[False, True],
            ascending=[e == "minimize" for e in directions],
        )
        .reset_index(drop=True)
    )
    return df


def _optuna_optimize_worker(
    objective,
    directions,
    metric_names,
    storage,
    study_name,
    sampler=None,
    pruner=None,
    n_trials=None,
    timeout=None,
    num_threads=1,
    catch=(),
    callbacks=None,
    gc_after_trial=False,
    show_progress_bar=False,
):
    study = optuna.create_study(
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
        load_if_exists=True,
        directions=directions,
    )
    study.set_metric_names(metric_names)
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=num_threads,
        catch=catch,
        callbacks=callbacks,
        gc_after_trial=gc_after_trial,
        show_progress_bar=show_progress_bar,
    )
    return None


def optuna_optimize_parallel(
    objective,
    directions,
    metric_names,
    storage,
    study_name,
    sampler=None,
    pruner=None,
    n_trials=None,
    timeout=None,
    num_processes=-1,
    num_threads=1,
    catch=(),
    callbacks=None,
    gc_after_trial=False,
    show_progress_bar=False,
    joblib_backend="multiprocessing",
):
    if num_processes < 0:
        assert num_processes == -1
        num_cpus = multiprocessing.cpu_count()
        logger.info("num_processes=%d, set to %d", num_processes, num_cpus)
        num_processes = num_cpus

    n_trials_per_process = n_trials
    if n_trials is not None:
        n_trials_per_process = math.ceil(n_trials / num_processes)
        logger.info(
            "n_trials=%d with num_process=%d --> %d trials per process",
            n_trials,
            num_processes,
            n_trials_per_process,
        )

    if num_processes > 1:
        logger.info("STARTING %d PROCESSES", num_processes)
        _ = Parallel(n_jobs=num_processes, backend=joblib_backend)(
            delayed(_optuna_optimize_worker)(
                objective,
                directions,
                metric_names,
                storage,
                study_name,
                sampler=sampler,
                pruner=pruner,
                n_trials=n_trials_per_process,
                timeout=timeout,
                num_threads=num_threads,
                catch=catch,
                callbacks=callbacks,
                gc_after_trial=gc_after_trial,
                show_progress_bar=show_progress_bar,
            )
            for _ in range(num_processes)
        )

        # return a study with all trials/results
        study = optuna.create_study(
            storage=storage,
            sampler=sampler,
            pruner=pruner,
            study_name=study_name,
            load_if_exists=True,
            directions=directions,
        )
        study.set_metric_names(metric_names)
        return study
    else:
        return _optuna_optimize_worker(
            objective,
            directions,
            metric_names,
            storage,
            study_name,
            sampler=sampler,
            pruner=pruner,
            n_trials=n_trials,
            timeout=timeout,
            num_threads=num_threads,
            catch=catch,
            callbacks=callbacks,
            gc_after_trial=gc_after_trial,
            show_progress_bar=show_progress_bar,
        )

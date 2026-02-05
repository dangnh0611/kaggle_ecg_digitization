import logging
import os

from hydra.core.config_search_path import ConfigSearchPath
from hydra.core.plugins import Plugins
from hydra.plugins.search_path_plugin import SearchPathPlugin

import yagm

logger = logging.getLogger(__name__)


def register_omegaconf_resolvers():
    from ast import literal_eval

    from omegaconf import OmegaConf

    def _no_slash(s, replace="|"):
        return s.replace("/", replace)

    OmegaConf.register_new_resolver(
        "_no_slash", _no_slash, replace=False, use_cache=False
    )
    OmegaConf.register_new_resolver(
        "_eval", lambda x: eval(x), replace=False, use_cache=False
    )
    OmegaConf.register_new_resolver(
        "_lit_eval", lambda x: literal_eval(x), replace=False, use_cache=False
    )
    OmegaConf.register_new_resolver(
        "_or", lambda x, y: x or y, replace=False, use_cache=False
    )
    OmegaConf.register_new_resolver(
        "_extend", lambda x, y: x.extend(y), replace=False, use_cache=False
    )
    OmegaConf.register_new_resolver(
        "_fname",
        lambda x: os.path.join(
            os.path.dirname(x),
            (
                os.path.basename(x)[:250] + "_ETC_"
                if len(os.path.basename(x)) > 250
                else os.path.basename(x)
            ),
        ),
        replace=False,
        use_cache=False,
    ),
    OmegaConf.register_new_resolver("_len", len, replace=False, use_cache=False)


class YAGMSearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        new_search_paths = []

        # From YAGM ROOT DIR
        root_config_dir = yagm.ROOT_CONFIGS_DIR
        assert "base" in os.listdir(root_config_dir)
        project_configs_dir = os.path.join(root_config_dir, "projects")
        if os.path.isdir(project_configs_dir):
            project_names = sorted(os.listdir(project_configs_dir))
            for project_name in project_names:
                new_search_path = os.path.join(project_configs_dir, project_name)
                new_search_paths.append(new_search_path)

        # From ENVIRONMENT VARIABLE
        HYDRA_SEARCH_PATH = [
            e for e in os.environ.get("HYDRA_SEARCH_PATH", "").split(":") if len(e)
        ]
        for p in HYDRA_SEARCH_PATH:
            if not os.path.isdir(p):
                logger.warning(
                    "Path %s is in HYDRA_SEARCH_PATH, but is not a directory. Skip!"
                )
            else:
                p = os.path.abspath(p)
                new_search_paths.extend([os.path.abspath(p) for p in HYDRA_SEARCH_PATH])

        # From Current Directory
        cwd = os.getcwd()
        for posible_subdir in ["configs", "config", "cfgs", "cfg"]:
            cwd_config_dir = os.path.join(cwd, posible_subdir)
            if os.path.isdir(cwd_config_dir):
                new_search_paths.append(cwd_config_dir)

        print(
            "The following directories will be added to Hydra search path:",
            new_search_paths,
        )
        for new_search_path in new_search_paths:
            search_path.append(provider="yagm-searchpath-plugin", path=new_search_path)
            print(f"Added {new_search_path} to Hydra search path")


def register_hydra_plugins() -> None:
    """Hydra users should call this function before invoking @hydra.main"""
    Plugins.instance().register(YAGMSearchPathPlugin)


def init_hydra() -> None:
    register_omegaconf_resolvers()
    register_hydra_plugins()

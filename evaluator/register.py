import importlib
import os
import pkgutil
import sys
from typing import Any, Callable


class DuplicateRegistrationError(Exception):
    """Raised when a method/function is registered more than once."""

    pass


class MetricRegister:
    def __init__(self):
        self._registry: dict[str, Callable[..., Any]] = {}

    def register(self, name: str | None = None) -> Callable[..., Any]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            reg_name = name or func.__name__

            if reg_name in self._registry:
                raise DuplicateRegistrationError(
                    f"Cannot register '{reg_name}' from {func.__module__}. "
                    f"A method with this name is already registered."
                )

            self._registry[reg_name] = func
            return func

        return decorator

    def get_registered_methods(self) -> dict[str, Callable[..., Any]]:
        return self._registry.copy()

    def discover_modules(self, package_dir: str, package_name: str):
        package_root = os.path.dirname(package_dir)
        if package_root not in sys.path:
            sys.path.insert(0, package_root)

        for _, module_name, is_pkg in pkgutil.walk_packages([package_dir]):
            full_name = f"{package_name}.{module_name}" if package_name else module_name
            try:
                importlib.import_module(full_name)
            except Exception as e:
                print(f"Failed to import module {full_name}: {e}")


_REGISTER = MetricRegister()
REGISTER_METRIC = _REGISTER.register
get_metrics = _REGISTER.get_registered_methods


def initialize_metrics():
    """Discover metrics and register one foreground-crop variant per whole-image metric."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    package_name = "evaluator.metrics"
    package_dir = os.path.join(base_dir, "metrics")
    _REGISTER.discover_modules(package_dir, package_name)
    from evaluator.metrics.crop import crop_metric

    metrics = get_metrics()
    for name, metric in metrics.items():
        if not name.endswith(("-FG", "-BG", "-CROP")) and f"{name}-CROP" not in metrics:
            REGISTER_METRIC(f"{name}-CROP")(crop_metric(metric))

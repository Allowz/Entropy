from __future__ import annotations

import importlib
import inspect
import pkgutil
from pathlib import Path
from typing import Any

from entropyos.plugins.base import (
    CompressorPlugin, EvaluatorPlugin, MemoryPlugin,
    RetrievalPlugin, ScorerPlugin,
)


class PluginRegistry:
    def __init__(self):
        self.scorers: dict[str, ScorerPlugin] = {}
        self.compressors: dict[str, CompressorPlugin] = {}
        self.evaluators: dict[str, EvaluatorPlugin] = {}
        self.retrieval: dict[str, RetrievalPlugin] = {}
        self.memory: dict[str, MemoryPlugin] = {}

    def register_scorer(self, plugin: ScorerPlugin) -> None:
        self.scorers[plugin.name()] = plugin

    def register_compressor(self, plugin: CompressorPlugin) -> None:
        self.compressors[plugin.name()] = plugin

    def register_evaluator(self, plugin: EvaluatorPlugin) -> None:
        self.evaluators[plugin.name()] = plugin

    def register_retrieval(self, plugin: RetrievalPlugin) -> None:
        self.retrieval[plugin.name()] = plugin

    def register_memory(self, plugin: MemoryPlugin) -> None:
        self.memory[plugin.name()] = plugin

    def discover(self, paths: list[str] | None = None) -> None:
        search_paths = paths or []
        for importer, modname, ispkg in pkgutil.iter_modules(search_paths):
            try:
                module = importlib.import_module(modname)
                self._scan_module(module)
            except Exception:
                pass

    def _scan_module(self, module) -> None:
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if obj is ScorerPlugin or obj is CompressorPlugin or obj is EvaluatorPlugin:
                continue
            if obj is RetrievalPlugin or obj is MemoryPlugin:
                continue
            try:
                if issubclass(obj, ScorerPlugin) and not inspect.isabstract(obj):
                    instance = obj()
                    self.register_scorer(instance)
                elif issubclass(obj, CompressorPlugin) and not inspect.isabstract(obj):
                    instance = obj()
                    self.register_compressor(instance)
                elif issubclass(obj, EvaluatorPlugin) and not inspect.isabstract(obj):
                    instance = obj()
                    self.register_evaluator(instance)
                elif issubclass(obj, RetrievalPlugin) and not inspect.isabstract(obj):
                    instance = obj()
                    self.register_retrieval(instance)
                elif issubclass(obj, MemoryPlugin) and not inspect.isabstract(obj):
                    instance = obj()
                    self.register_memory(instance)
            except Exception:
                pass


plugin_registry = PluginRegistry()

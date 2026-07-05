"""Pipeline-shared DataLoaderFactory: one loader pool per run, trainer ctor stops re-spawning workers."""

import pickle

import torch

from conftest import MockDataProviderFactory, MockPipeline

from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.model_training.training_utilities import BasicClassificationLoss


def _pipeline(num_workers=0):
    return MockPipeline(config={"num_workers": num_workers, "device": "cpu"})


class _TinyModel(torch.nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.linear = torch.nn.Linear(64, num_classes)

    def forward(self, x):
        return self.linear(x.reshape(x.shape[0], -1))


class TestForPipelineReturnsSharedFactory:
    def test_same_pipeline_gets_the_same_factory_instance(self):
        pipeline = _pipeline()
        first = DataLoaderFactory.for_pipeline(pipeline)
        second = DataLoaderFactory.for_pipeline(pipeline)
        assert first is second

    def test_different_pipelines_get_different_factories(self):
        one = DataLoaderFactory.for_pipeline(_pipeline())
        two = DataLoaderFactory.for_pipeline(_pipeline())
        assert one is not two

    def test_shared_factory_owns_loaders(self):
        factory = DataLoaderFactory.for_pipeline(_pipeline())
        assert factory.owns_loaders() is True

    def test_directly_constructed_factory_does_not_own_loaders(self):
        factory = DataLoaderFactory(MockDataProviderFactory(), num_workers=0)
        assert factory.owns_loaders() is False

    def test_num_workers_change_rebuilds_the_cached_factory(self):
        pipeline = _pipeline(num_workers=0)
        first = DataLoaderFactory.for_pipeline(pipeline)
        pipeline.config["num_workers"] = 2
        second = DataLoaderFactory.for_pipeline(pipeline)
        assert second is not first
        assert second._num_workers == 2


class TestSharedLoaderCache:
    def test_same_split_and_batch_size_reuses_the_loader(self):
        factory = DataLoaderFactory.for_pipeline(_pipeline())
        provider = factory.create_data_provider()
        first = factory.create_training_loader(4, provider)
        second = factory.create_training_loader(4, provider)
        assert first is second

    def test_distinct_batch_sizes_get_distinct_loaders(self):
        factory = DataLoaderFactory.for_pipeline(_pipeline())
        provider = factory.create_data_provider()
        assert factory.create_training_loader(4, provider) is not (
            factory.create_training_loader(2, provider)
        )

    def test_distinct_splits_get_distinct_loaders(self):
        factory = DataLoaderFactory.for_pipeline(_pipeline())
        provider = factory.create_data_provider()
        train = factory.create_training_loader(4, provider)
        val = factory.create_validation_loader(4, provider)
        test = factory.create_test_loader(4, provider)
        assert len({id(train), id(val), id(test)}) == 3

    def test_unshared_factory_builds_fresh_loaders_every_call(self):
        factory = DataLoaderFactory(MockDataProviderFactory(), num_workers=0)
        provider = factory.create_data_provider()
        assert factory.create_training_loader(4, provider) is not (
            factory.create_training_loader(4, provider)
        )

    def test_close_cached_loaders_empties_the_pool(self):
        factory = DataLoaderFactory.for_pipeline(_pipeline())
        provider = factory.create_data_provider()
        loader = factory.create_training_loader(4, provider)
        factory.close_cached_loaders()
        assert factory.create_training_loader(4, provider) is not loader


class TestSharedFactoryPickleSafety:
    def test_pickle_drops_live_loader_and_eval_caches(self):
        factory = DataLoaderFactory.for_pipeline(_pipeline())
        provider = factory.create_data_provider()
        factory.create_training_loader(4, provider)
        factory.put_eval_cache(("val", 4, "cpu", None), [("x", "y")])
        clone = pickle.loads(pickle.dumps(factory))
        assert clone._loader_cache == {}
        assert clone._eval_cache == {}
        assert clone.owns_loaders() is True


class TestTrainerCloseWithSharedFactory:
    def test_close_leaves_shared_loaders_alive(self):
        pipeline = _pipeline(num_workers=0)
        factory = DataLoaderFactory.for_pipeline(pipeline)
        trainer = BasicTrainer(
            _TinyModel(), "cpu", factory, BasicClassificationLoss()
        )
        shared_train_loader = trainer.train_loader
        trainer.close()
        assert trainer.train_loader is None
        # The pool still hands out the same live loader to the next trainer.
        provider = factory.create_data_provider()
        assert factory.create_training_loader(
            provider.get_training_batch_size(), provider
        ) is shared_train_loader

    def test_close_still_shuts_down_unshared_loaders(self, monkeypatch):
        import mimarsinan.model_training.basic_trainer as bt_mod

        calls = []
        monkeypatch.setattr(
            bt_mod, "shutdown_data_loader", lambda loader: calls.append(loader)
        )
        factory = DataLoaderFactory(MockDataProviderFactory(), num_workers=0)
        trainer = BasicTrainer(
            _TinyModel(), "cpu", factory, BasicClassificationLoss()
        )
        trainer.close()
        assert len(calls) == 3

    def test_two_trainers_on_the_shared_pool_reuse_loaders(self):
        pipeline = _pipeline(num_workers=0)
        factory = DataLoaderFactory.for_pipeline(pipeline)
        a = BasicTrainer(_TinyModel(), "cpu", factory, BasicClassificationLoss())
        b = BasicTrainer(_TinyModel(), "cpu", factory, BasicClassificationLoss())
        assert a.train_loader is b.train_loader
        assert a.validation_loader is b.validation_loader
        assert a.test_loader is b.test_loader


class TestSharedValCache:
    def test_val_cache_content_is_shared_and_cursors_are_independent(self):
        pipeline = _pipeline(num_workers=0)
        factory = DataLoaderFactory.for_pipeline(pipeline)
        a = BasicTrainer(_TinyModel(), "cpu", factory, BasicClassificationLoss())
        b = BasicTrainer(_TinyModel(), "cpu", factory, BasicClassificationLoss())
        batches_a = [x for x, _ in a.iter_validation_batches(2)]
        assert a._gpu_val_cache is not None
        assert getattr(b, "_gpu_val_cache", None) is None
        batches_b = [x for x, _ in b.iter_validation_batches(2)]
        assert b._gpu_val_cache is a._gpu_val_cache
        # b starts at cursor 0 regardless of a's progress: identical sequences.
        for xa, xb in zip(batches_a, batches_b):
            assert torch.equal(xa, xb)

    def test_val_cache_key_honors_max_batches_limit(self):
        pipeline = _pipeline(num_workers=0)
        factory = DataLoaderFactory.for_pipeline(pipeline)
        full = BasicTrainer(_TinyModel(), "cpu", factory, BasicClassificationLoss())
        list(full.iter_validation_batches(1))
        limited = BasicTrainer(_TinyModel(), "cpu", factory, BasicClassificationLoss())
        limited._val_cache_max_batches = 1
        list(limited.iter_validation_batches(1))
        assert limited._gpu_val_cache is not full._gpu_val_cache
        assert len(limited._gpu_val_cache) == 1

    def test_shared_val_cache_matches_unshared_eval_basis_exactly(self):
        """The gate-grade probe set must be bit-identical with and without sharing."""
        provider_factory = MockDataProviderFactory()
        pipeline = MockPipeline(
            config={"num_workers": 0, "device": "cpu"},
            data_provider_factory=provider_factory,
        )
        shared_factory = DataLoaderFactory.for_pipeline(pipeline)
        unshared_factory = DataLoaderFactory(provider_factory, num_workers=0)
        shared = BasicTrainer(
            _TinyModel(), "cpu", shared_factory, BasicClassificationLoss()
        )
        unshared = BasicTrainer(
            _TinyModel(), "cpu", unshared_factory, BasicClassificationLoss()
        )
        for (xs, ys), (xu, yu) in zip(
            shared.iter_validation_batches(3), unshared.iter_validation_batches(3)
        ):
            assert torch.equal(xs, xu)
            assert torch.equal(ys, yu)

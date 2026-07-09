"""PipelineSession.finish() releases pooled data-loader resources."""

from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
from mimarsinan.pipelining.session import PipelineSession

from conftest import MockPipeline


class _Reporter:
    def __init__(self):
        self.finished = False

    def finish(self):
        self.finished = True


def _bare_session(pipeline, reporter):
    session = object.__new__(PipelineSession)
    session.pipeline = pipeline
    session.reporter = reporter
    return session


class TestFinishClosesPooledLoaders:
    def test_finish_closes_shared_factory_loaders(self):
        pipeline = MockPipeline(config={"num_workers": 0, "device": "cpu"})
        factory = DataLoaderFactory.for_pipeline(pipeline)
        provider = factory.create_data_provider()
        factory.create_training_loader(4, provider)
        assert factory._loader_cache

        reporter = _Reporter()
        _bare_session(pipeline, reporter).finish()

        assert reporter.finished
        assert not factory._loader_cache

    def test_finish_without_factory_still_finishes_reporter(self):
        reporter = _Reporter()
        _bare_session(MockPipeline(), reporter).finish()
        assert reporter.finished

    def test_reporter_failure_still_closes_loaders(self):
        class _ExplodingReporter:
            def finish(self):
                raise RuntimeError("reporter down")

        pipeline = MockPipeline(config={"num_workers": 0, "device": "cpu"})
        factory = DataLoaderFactory.for_pipeline(pipeline)
        provider = factory.create_data_provider()
        factory.create_training_loader(4, provider)

        _bare_session(pipeline, _ExplodingReporter()).finish()
        assert not factory._loader_cache

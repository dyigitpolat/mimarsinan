import sys
sys.path.append('./src')

from src.init import init
from src.main import main, run_pipeline_from_config

if __name__ == "__main__":
    init()
    if len(sys.argv) >= 2 and sys.argv[1] == "--ui":
        from mimarsinan.gui.data_collector import DataCollector
        from mimarsinan.gui.server import start_server

        collector = DataCollector()
        start_server(collector, run_config_fn=run_pipeline_from_config)
        try:
            input("Press Enter to exit...\n")
        except (KeyboardInterrupt, EOFError):
            pass
    else:
        main()

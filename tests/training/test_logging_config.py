import logging

from optastra.training.logging_config import ColorFormatter, setup_logging


def test_setup_logging_writes_file_and_configures_logger_tree(tmp_path):
    logger = setup_logging(tmp_path, filename="optastra.log", color=False)

    child = logging.getLogger("optastra.train")
    child.info("hello world")

    log_path = tmp_path / "optastra.log"
    assert log_path.exists()
    contents = log_path.read_text()
    assert "hello world" in contents

    assert logger.name == "optastra"


def test_color_formatter_wraps_known_levels():
    formatter = ColorFormatter("%(levelname)s:%(message)s")
    record = logging.LogRecord("optastra.train", logging.ERROR, __file__, 1, "boom", (), None)

    formatted = formatter.format(record)
    assert "boom" in formatted
    assert "\033[31m" in formatted
    assert formatted.startswith("\033[31m[")
    assert "| optastra.train ERROR]" in formatted
    assert formatted.endswith("\033[0m: boom")
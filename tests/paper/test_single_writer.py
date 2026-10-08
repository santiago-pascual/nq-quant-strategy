import pytest

from src.paper.single_writer import PaperWriterLock, WriterAlreadyActive


def test_output_directory_lock_prevents_second_writer_and_releases(tmp_path):
    path = tmp_path / "paper_writer.lock"
    first = PaperWriterLock(path).acquire()
    try:
        with pytest.raises(WriterAlreadyActive):
            PaperWriterLock(path).acquire()
    finally:
        first.release()

    with PaperWriterLock(path):
        pass

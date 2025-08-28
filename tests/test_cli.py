import pytest
from nitropulse.core import main


def test_phenology_command(mocker):
    """
    Tests that `nitropulse phenology` calls the correct underlying function.
    We mock the target function to isolate the test to only the CLI parsing logic.
    """
    # Mock the function that does the actual work to prevent it from running.
    mock_phenology_command = mocker.patch('nitropulse.core.phenology_command')

    # Simulate running 'nitropulse phenology' from the command line by patching sys.argv.
    mocker.patch('sys.argv', ['nitropulse', 'phenology'])

    # Call the main CLI entrypoint.
    main()

    # Assert that our mocked function was called exactly once.
    mock_phenology_command.assert_called_once()


def test_run_command_help(capsys, mocker):
    """
    Tests that `nitropulse run --help` prints help text and exits cleanly.
    """
    # Mock sys.argv to simulate the command.
    mocker.patch('sys.argv', ['nitropulse', 'run', '--help'])

    # The --help action in argparse causes a SystemExit. We expect this.
    with pytest.raises(SystemExit) as e:
        main()

    # Check that the exit code is 0 (successful exit).
    assert e.value.code == 0

    # Use the capsys fixture to capture output printed to stdout.
    captured = capsys.readouterr()

    # Check if some expected help text is in the output.
    assert "Runs the complete workflow" in captured.out
    assert "--roi-asset-id" in captured.out


def test_download_risma_with_args(mocker):
    """
    Tests that `nitropulse download-risma` passes arguments correctly to the target function.
    """
    mock_download_risma = mocker.patch('nitropulse.core.download_risma_command')

    test_argv = ['nitropulse', 'download-risma', '--stations', 'RISMA_MB1', '--start-date', '2022-01-01']
    mocker.patch('sys.argv', test_argv)

    main()

    mock_download_risma.assert_called_once()
    call_args = mock_download_risma.call_args[0][0]
    assert call_args.stations == ['RISMA_MB1']
    assert call_args.start_date == '2022-01-01'
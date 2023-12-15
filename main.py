from argument import args
from synthesize.main import main as synth_main
from validation.main import (
    main as valid_main,
)  # The relabel and validation are combined here for fast experiment
import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    synth_main(args)
    valid_main(args)

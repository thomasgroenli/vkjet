"""`python -m vkjet.tests [-v] [name ...]`

Without names, discovers every test_*.py in this package. A name is a module
(`test_eqrow`), a class (`test_eqrow.TestEqRow`) or a method
(`test_eqrow.TestEqRow.test_loss`), resolved inside vkjet.tests.
"""
import os
import sys
import unittest


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    verbosity = 2 if "-v" in argv else 1
    names = [a for a in argv if not a.startswith("-")]
    loader = unittest.TestLoader()
    if names:
        suite = loader.loadTestsFromNames([f"{__package__}.{n}" for n in names])
    else:
        here = os.path.dirname(os.path.abspath(__file__))
        top = os.path.dirname(os.path.dirname(here))
        suite = loader.discover(here, pattern="test_*.py", top_level_dir=top)
    result = unittest.TextTestRunner(verbosity=verbosity).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())

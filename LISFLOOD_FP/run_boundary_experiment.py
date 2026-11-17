"""JSON-stable CLI for starting/resuming the frozen example-56 experiment.

Normalize scenario tuples before protocol comparison with a saved JSON document.
The numerical driver and its recorded source hashes remain unchanged.
"""
import calibrate_boundaries56 as experiment


if __name__=='__main__':
    experiment.SCENARIOS={key:list(value) for key,value in experiment.SCENARIOS.items()}
    experiment.main()

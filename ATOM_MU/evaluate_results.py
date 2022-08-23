import sys
import os
sys.path.append(os.path.join('../utils'))
import neuron.data as data


def evaluate(dataset, trackers=None):
    if dataset == 'lasot':
        evaluators = [data.EvaluatorLaSOT()]
    elif dataset == 'tlp':
        evaluators = [data.EvaluatorTLP()]
    elif dataset=='otb':
        evaluators=[data.EvaluatorOTB()]
    elif dataset=='uav123':
        evaluators=[data.EvaluatorUAV123()]
    elif dataset=='votlt18':
        evaluators=[data.EvaluatorVOT(version='2018LT')]
    elif dataset=='votlt19':
        evaluators=[data.EvaluatorVOT(version='2019LT')]
    if trackers is None:
        # trackers = os.listdir('./results')
        trackers=['ATOM_MU']
    for e in evaluators:
        e.report(trackers, dataset=dataset, plot_curves=True)


if __name__ == '__main__':
    evaluate(dataset='lasot')



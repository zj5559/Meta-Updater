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
    if trackers is None:
        # trackers = os.listdir('./results')
        trackers=['RTMD','RTMD_MU_33','RTMD_fusion_v4_reverse_42','RTMD_fusion_v4_reverse_42_lof2.5','RTMD_fusion_v4_reverse_42_lof2.1',
                  'RTMD_fusion_v4_reverse_42_lof1.8','RTMD_fusion_v4_reverse_42_lof2.7','RTMD_fusion_v4_reverse_42_lof3.0','RTMD_fusion_v4_reverse_42_lof3.5']
    for e in evaluators:
        e.report(trackers, dataset=dataset, plot_curves=True)


if __name__ == '__main__':
    evaluate(dataset='uav123')
    # evaluate(dataset='tlp')



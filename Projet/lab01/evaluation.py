from typing import List

def precision_recall(expected_results: List[bool], actual_results: List[bool]) -> (float, float):
    """Compute the precision and recall of a series of predictions

    Parameters
    ----------
        expected_results : List[bool]
            The true results, that is the results that the predictor
            should have find.
        actual_results : List[bool]
            The predicted results, that have to be evaluated.

    Returns
    -------
        float
            The precision of the predicted results.
        float
            The recall of the predicted results.
    """
    TP = 0 ; FP =0 ; FN = 0
    for i in range(len(expected_results)):
        if actual_results[i] :
            if expected_results[i]:
                TP +=1
            else:
                FP+=1
        elif expected_results[i]:
            FN+=1
    precision = TP/(TP +FP)
    recall = TP/(TP+FN)
    return(precision, recall)
        
    raise NotImplementedError('Implement this method for Question 3')

def F1_score(expected_results: List[bool], actual_results: List[bool]) -> float:
    """Compute the F1-score of a series of predictions

    Parameters
    ----------
        expected_results : List[bool]
            The true results, that is the results that the predictor
            should have find.
        actual_results : List[bool]
            The predicted results, that have to be evaluated.

    Returns
    -------
        float
            The F1-score of the predicted results.
    """
    (p, r) = precision_recall(expected_results, actual_results)
    return 2*r*p/(r+p)
    raise NotImplementedError('Implement this method for Question 3')

import numpy as np


def get_effective_diameter(ground_truth_mu, pixel_width_mm):
    '''
    effective diameter defined in AAPM TG204: https://www.aapm.org/pubs/reports/RPT_204.pdf
    '''
    A = np.sum(ground_truth_mu > -1000)*pixel_width_mm**2
    return 2*np.sqrt(A/np.pi)


def cosine_similarity(a, b):
    a = a.ravel()
    b = b.ravel()
    a = a.astype(float)
    b = b.astype(float)
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

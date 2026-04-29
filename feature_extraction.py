import numpy as np
import cv2
from scipy import ndimage
from scipy.signal import convolve2d
def compute_lpq(image, winSize=3, freqestim=1, mode='im'):

    rho = 0.90

    STFTalpha = 1 / winSize  # alpha in STFT approaches (for Gaussian derivative alpha=1)
    sigmaS = (winSize - 1) / 4  # Sigma for STFT Gaussian window (applied if freqestim==2)
    sigmaA = 8 / (winSize - 1)  # Sigma for Gaussian derivative quadrature filters (applied if freqestim==3)

    convmode = 'valid'  # Compute descriptor responses only on part that have full neigborhood. Use 'same' if all pixels are included (extrapolates np.image with zeros).

    img = np.float64(image)  # Convert np.image to double
    r = (winSize - 1) / 2  # Get radius from window size
    x = np.arange(-r, r + 1)[np.newaxis]  # Form spatial coordinates in window

    if freqestim == 1:  # STFT uniform window
        #  Basic STFT filters
        w0 = np.ones_like(x)
        w1 = np.exp(-2 * np.pi * x * STFTalpha * 1j)
        w2 = np.conj(w1)

    ## Run filters to compute the frequency response in the four points. Store np.real and np.imaginary parts separately
    # Run first filters
    filterResp1 = convolve2d(convolve2d(img, w0.T, convmode), w1, convmode)
    filterResp2 = convolve2d(convolve2d(img, w1.T, convmode), w0, convmode)
    filterResp3 = convolve2d(convolve2d(img, w1.T, convmode), w1, convmode)
    filterResp4 = convolve2d(convolve2d(img, w1.T, convmode), w2, convmode)

    # Initilize frequency domain matrix for four frequency coordinates (np.real and np.imaginary parts for each frequency).
    freqResp = np.dstack([filterResp1.real, filterResp1.imag,
                          filterResp2.real, filterResp2.imag,
                          filterResp3.real, filterResp3.imag,
                          filterResp4.real, filterResp4.imag])

    ## Perform quantization and compute LPQ codewords
    inds = np.arange(freqResp.shape[2])[np.newaxis, np.newaxis, :]
    LPQdesc = ((freqResp > 0) * (2 ** inds)).sum(2)
    return np.mean(LPQdesc, axis=0)
def compute_ldp(image):
    na = np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]])
    wa = np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]])
    sa = np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]])
    nea = np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]])
    nwa = np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]])
    sea = np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]])
    swa = np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]])
    ka = np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]])
    e_k = ndimage.convolve(image, ka, mode='nearest', cval=0.0)
    n_k = ndimage.convolve(image, na, mode='nearest', cval=0.0)
    s_k = ndimage.convolve(image, sa, mode='nearest', cval=0.0)
    w_k = ndimage.convolve(image, wa, mode='nearest', cval=0.0)
    ne_k = ndimage.convolve(image, nea, mode='nearest', cval=0.0)
    nw_k = ndimage.convolve(image, nwa, mode='nearest', cval=0.0)
    se_k = ndimage.convolve(image, sea, mode='nearest', cval=0.0)
    sw_k = ndimage.convolve(image, swa, mode='nearest', cval=0.0)

    ldp_mat = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            lst = [se_k[i][j], s_k[i][j], sw_k[i][j], w_k[i][j], nw_k[i][j], n_k[i][j], ne_k[i][j], e_k[i][j]]
            l = [abs(h) for h in lst]
            marr = np.argsort(l)
            marr1 = marr[::-1]
            binary = np.zeros(8, dtype="uint8")
            binary[marr1[0]] = 1
            binary[marr1[1]] = 1
            binary[marr1[2]] = 1
            d_no = binary[0] * 2 ** 7 + binary[1] * 2 ** 6 + binary[2] * 2 ** 5 + binary[3] * 2 ** 4 + binary[
                4] * 2 ** 3 + binary[5] * 2 ** 2 + binary[6] * 2 ** 1 + binary[7] * 2 ** 0
            ldp_mat[i][j] = d_no

    return np.mean(ldp_mat, axis=0)


def color_moments(image):
    moments = cv2.moments(image)
    colorFeatures = np.array(list(moments.values()))
    return colorFeatures


def ShapeFeature(image):
    image = image.astype('uint8')
    # Thresholding
    _, binary_image = cv2.threshold(image, 15, 29, cv2.THRESH_BINARY)
    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shape_feat = []
    # Calculate contour-based features for each contour
    for contour in contours:
        # Area
        area = cv2.contourArea(contour)
        if area == 0.0:
            continue
        # Perimeter
        perimeter = cv2.arcLength(contour, True)
        # Compactness
        compactness = (perimeter ** 2) / (4 * 3.1415 * area)
        feat = np.concatenate([np.atleast_1d(area, perimeter, compactness)])
        shape_feat.extend(feat.flatten())
        return shape_feat


def feature_extraction(image):
    # Texture Features
    lpq_feat = compute_lpq(image)
    ldp_feat = compute_ldp(image)
    texture_feat = np.concatenate((lpq_feat, ldp_feat))
    # Color Based Feature
    color_moment = color_moments(image)
    # Shape Based Feature
    shape_feat = ShapeFeature(image)
    if shape_feat is None:
        shape_feat = np.zeros(3)
    feature = np.concatenate((texture_feat, color_moment, shape_feat))
    return feature
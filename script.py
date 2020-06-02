import numpy as np
import sys
import json


def estimate(hypocenter, stations, initial=None):
    if initial is None:
        initial = np.mean(np.asarray(hypocenter), axis=1)
        initial[2] = -1000
    if len(stations) < 3:
        raise ValueError("There have to be at least 3 stations")
    # actual hypocenter location
    xh = hypocenter[0]
    yh = hypocenter[1]
    zh = hypocenter[2]

    # velocity of P-Wave (m/s)
    vavg = 3500

    x = np.asarray([st[0] for st in stations])
    y = np.asarray([st[1] for st in stations])
    z = np.asarray([st[2] for st in stations])

    # adding some noise to make results more accurate (as velocity of P-Wave can differ in different kind of matter)
    noise = 5e-3

    tsint = np.sqrt(np.power(xh - x, 2) + np.power(yh - y, 2) + np.power(zh - z, 2)) / vavg

    ti = tsint + np.random.randn(len(tsint)) * noise

    # initial hypocenter (used for further estimation)
    x0 = initial[0]
    y0 = initial[1]
    z0 = initial[2]

    iterations = 0
    hypocenter_temporal = []
    np.append(hypocenter_temporal, [x0, y0, z0])

    error = [float('Inf')]
    xx = []
    yy = []
    zz = []

    # How big difference is between errors received in current and previous iteration
    err_threshold = 1e-10
    tcal_ = []
    while True:
        # time, after which P-wave was should be detected if hypocenter is in location (x0, y0, z0)
        tcal = np.sqrt(np.power(x0 - x, 2) + np.power(y0 - y, 2) + np.power(z0 - z, 2)) / vavg
        # divider of derivatives
        vavg_distance = vavg * np.sqrt(np.power(x0 - x, 2) + np.power(y0 - y, 2) + np.power(z0 - z, 2))
        dtdx0 = (x0 - x) / vavg_distance
        dtdy0 = (y0 - y) / vavg_distance
        dtdz0 = (z0 - z) / vavg_distance

        # error measured in seconds
        err = np.sqrt(np.power(np.mean(ti - tcal), 2))

        # Jacoby matrix (transposion is used to have x, y and z values respectively as columns, not rows)
        J = np.block([[dtdx0], [dtdy0], [dtdz0]]).T
        Jt = J.T

        # difference of model
        dm = np.linalg.inv(Jt.dot(J)).dot(Jt) * (ti.T - tcal.T)

        # overall change of x, y, z
        dm_ = np.sum(dm, axis=1)
        x0 = x0 + dm_[0]
        y0 = y0 + dm_[1]
        z0 = z0 + dm_[2]
        xx.append(x0)
        yy.append(y0)
        zz.append(z0)
        tcal_.append(tcal)
        iterations = iterations + 1
        error.append(err)
        # loop is broken if difference between errors is smaller than threshold
        if np.abs(err - error[len(error) - 2]) < err_threshold:
            error = error[1:]
            break

    return [xx[len(xx) - 1], yy[len(yy) - 1], zz[len(zz) - 1]], error[len(error) - 1], [xx, yy, zz], error


def main():
    output_path = ''
    if len(sys.argv) < 2:
        raise IOError('Program needs JSON file with data')
    elif len(sys.argv) < 3:
        output_path = 'output.json'
    else:
        output_path = sys.argv[2]
    output = {}
    with open(sys.argv[1], 'r') as f:
        data = json.load(f)
        data = data["data"]

        output["output"] = []
        for i in range(len(data)):
            result, error, res_hist, err_hist = estimate(data[i]["hypocenter"], data[i]["stations"], data[i]["initial"])
            output["output"].append({
                'result': result,
                'error': error,
                'history': {
                    'result': res_hist,
                    'error': err_hist
                }
            })
    with open(output_path, 'w+') as out:
        json.dump(output, out)


if __name__ == '__main__':
    main()

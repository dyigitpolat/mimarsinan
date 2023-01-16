# regions: 2, 8
# center: -0.15, 0.15
# exp: 0.5, 2
def get_region_borders(regions, center, exp, length):
    if(regions == 1):
        return [0, length]

    borders = [ 2 * (x/regions) - 1.0 for x in range(regions + 1)]
    
    for i in range(1, regions):
        if(borders[i] < 0):
            borders[i] = -(abs(borders[i])**exp)
        else:
            borders[i] = borders[i]**exp

        borders[i] += center
        borders[i] = max(borders[i], -1.0)
        borders[i] = min(borders[i], 1.0)
    
    borders = [(x+1) / 2 for x in borders]

    borders = [round(v * length) for v in borders]
    for i in range(len(borders)//2):
        if(borders[i] >= borders[i+1]):
            borders[i+1] = borders[i] + 1
    
    for i in range(len(borders) - 1, len(borders)//2, -1):
        if(borders[i] <= borders[i-1]):
            borders[i-1] = borders[i] - 1

    return borders
###loading all relevant packages/libraries
import pandas as pd
import numpy as np
from decimal import Decimal
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
import colorsys


###get exponent & mantissa

def fexp(number):
    (sign, digits, exponent) = Decimal(number).as_tuple()
    return len(digits) + exponent - 1

def fman(number):
    return Decimal(number).scaleb(-fexp(number)).normalize()


###create linear interpolated (Lab-space) color palette

def create_palette(start_rgb, end_rgb, n):

    ## convert start and end to a point in the given colorspace
    start = convert_color(start_rgb, LabColor).get_value_tuple()
    end = convert_color(end_rgb, LabColor).get_value_tuple()

    ## create a set of n points along start to end
    points = list(zip(*[np.linspace(start[i], end[i], n) for i in range(3)]))

    ## create a color for each point and convert back to rgb
    rgb = [convert_color(LabColor(*point), sRGBColor) for point in points]
    rgb = list(c.get_value_tuple() for c in rgb)

    #rgb_colors = np.maximum(np.minimum(rgb, [1, 1, 1]), [0, 0, 0])
    #rgb_colors = [max(min(i, (1, 1, 1)), (0, 0, 0)) for i in rgb]

    # rgb_colors = []
    # for color in rgb:
    #     c = list(color)
    #     for i in range(3):
    #         if c[i] > 1:
    #             c[i] = 2 - c[i]
    #         if c[i] < 0:
    #             c[i] *= -1
    #     rgb_colors.append(c)

    return rgb#_colors


###generate OMC-map from given colorscale and extrema

def generate_omc(min_value, max_value, colorscale = "viridis"):

    ##calculate max/min exponent
    max_exp = fexp(Decimal(str(max_value)))+1
    min_exp = fexp(Decimal(str(min_value)))

    ##number of colors needed
    nr_colors = max_exp - min_exp
    
    ##number of colors per seq scale
    nr_scale = int(1000/nr_colors)

    ##positions of searched colors in original cmap
    position_list_orig = list(np.linspace(0, 1, nr_colors))

    ##positions of colors for final cmap
    position_list_fin = list()
    for i in range(1,nr_colors + 1):
        position_list_fin = position_list_fin + list(np.linspace((i-1)/nr_colors, i/nr_colors, nr_scale))
        
    ##get cmap
    cmap = cm.get_cmap(colorscale)

    rgb_list = list()
    
    ##create sequential palettes for every new exponent
    for i in position_list_orig:
        color_start = cmap(i)[:-1]
        #color_start_rgb = sRGBColor(color_start[0], color_start[1], color_start[2])
        color_start_hsv = colorsys.rgb_to_hsv(cmap(i)[0], cmap(i)[1], cmap(i)[2])
        color_start_hsv_temp = [color_start_hsv[0], color_start_hsv[1]+0.1, color_start_hsv[2]+0.1]
        color_start_temp = colorsys.hsv_to_rgb(color_start_hsv_temp[0], color_start_hsv_temp[1], color_start_hsv_temp[2])
        color_start_rgb_temp = sRGBColor(color_start_temp[0], color_start_temp[1], color_start_temp[2])
        color_end_hsv = [color_start_hsv[0], color_start_hsv[1]-0.25, color_start_hsv[2]-0.25]
        color_end = colorsys.hsv_to_rgb(color_end_hsv[0], color_end_hsv[1], color_end_hsv[2])
        color_end_rgb = sRGBColor(color_end[0], color_end[1], color_end[2])
        palette = create_palette(color_start_rgb_temp, color_end_rgb, nr_scale)
        rgb_list.extend(palette)

    ##check for doubled colors
    #rgb_list_uniq = set(rgb_list)
    #if(len(rgb_list_uniq) == len(rgb_list)):
        #print("No doubled colors!")
    #else: 
        #print("Doubled colors!")# Generated Viridis-OMC color scale instead.")
        #return generate_omc(min_value, max_value, "viridis")


    ##create rgb dict
    cdict = dict()

    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[position_list_fin[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(position_list_fin))]
        cdict[col] = col_list
    
    ##create linear segmented colormap out of cdict
    colormap = cm.colors.LinearSegmentedColormap('colormap', segmentdata=cdict, N=1000)

    return (colormap, min_exp, max_exp)


###get color values by data value

def get_rgb(colormap, value, max_exp, min_exp):

    cmap = cm.get_cmap(colormap, 1000)

    value_mant = fman(value)
    value_exp = fexp(value)

    range_base = max_exp-min_exp
    range_value = (value_exp-min_exp)+(value_mant-1)/10

    position = float(range_value/range_base)
    
    return cmap(position)[:-1]

def get_hsv(colormap, value, max_exp, min_exp):

    rgb = get_rgb(colormap, value, max_exp, min_exp)

    hsv = colorsys.rgb_to_hsv(rgb[0], rgb[1], rgb[2])

    return hsv


###function to convert rgb to lab color

def rgb2lab(inputColor):

   num = 0
   RGB = [0, 0, 0]

   for value in inputColor :
       value = float(value) / 255

       if value > 0.04045 :
           value = ( ( value + 0.055 ) / 1.055 ) ** 2.4
       else :
           value = value / 12.92

       RGB[num] = value * 100
       num = num + 1

   XYZ = [0, 0, 0,]

   X = RGB [0] * 0.4124 + RGB [1] * 0.3576 + RGB [2] * 0.1805
   Y = RGB [0] * 0.2126 + RGB [1] * 0.7152 + RGB [2] * 0.0722
   Z = RGB [0] * 0.0193 + RGB [1] * 0.1192 + RGB [2] * 0.9505
   XYZ[ 0 ] = round( X, 4 )
   XYZ[ 1 ] = round( Y, 4 )
   XYZ[ 2 ] = round( Z, 4 )

   XYZ[ 0 ] = float( XYZ[ 0 ] ) / 95.047         
   XYZ[ 1 ] = float( XYZ[ 1 ] ) / 100.0          
   XYZ[ 2 ] = float( XYZ[ 2 ] ) / 108.883      

   num = 0
   for value in XYZ :

       if value > 0.008856 :
           value = value ** ( 0.3333333333333333 )
       else :
           value = ( 7.787 * value ) + ( 16 / 116 )

       XYZ[num] = value
       num = num + 1

   Lab = [0, 0, 0]

   L = ( 116 * XYZ[ 1 ] ) - 16
   a = 500 * ( XYZ[ 0 ] - XYZ[ 1 ] )
   b = 200 * ( XYZ[ 1 ] - XYZ[ 2 ] )

   Lab [ 0 ] = round( L, 4 )
   Lab [ 1 ] = round( a, 4 )
   Lab [ 2 ] = round( b, 4 )

   return Lab


##show cmap incl. greyscale

def grayscale_cmap(cmap):
    cmap = plt.cm.get_cmap(cmap, 1000)
    colors = cmap(np.arange(cmap.N))
    RGB_weight = [0.299, 0.587, 0.114]
    luminance = np.sqrt(np.dot(colors[:, :3] ** 2, RGB_weight))
    colors[:, :3] = luminance[:, np.newaxis]
        
    return LinearSegmentedColormap.from_list(cmap.name + "_gray", colors, cmap.N)
    
def view_colormap(cmap):
    cmap = plt.cm.get_cmap(cmap, 1000)
    colors = cmap(np.arange(cmap.N))
    
    cmap = grayscale_cmap(cmap)
    grayscale = cmap(np.arange(cmap.N))
    
    fig, ax = plt.subplots(2, figsize=(13, 8),
                           subplot_kw=dict(xticks=[], yticks=[]))
    ax[0].imshow([colors], extent=[0, 10, 0, 1])
    ax[1].imshow([grayscale], extent=[0, 10, 0, 1])
    fig.tight_layout()


##function for plotting rgb- and hsv-gradients

def show_gradients(colorscale):
    
    rgb = pd.DataFrame()
    hsv = pd.DataFrame()
    x = list()
    red = list()
    green = list()
    blue = list()
    h = list()
    s = list()
    v = list()

    ##get color values of color scale
    cmap = cm.get_cmap(colorscale, 1000)

    ##create rgb- and hsv-arrays
    for i in range(0,1001,1):
        j = i/1000
        x.append(j)
        red.append(cmap(j)[0])
        green.append(cmap(j)[1])
        blue.append(cmap(j)[2])
        color_hsv = colorsys.rgb_to_hsv(cmap(j)[0], cmap(j)[1], cmap(j)[2])
        h.append(color_hsv[0])
        s.append(color_hsv[1])
        v.append(color_hsv[2])

    rgb["x"] = x
    rgb["red"] = red
    rgb["green"] = green
    rgb["blue"] = blue

    hsv["x"] = x
    hsv["h"] = h
    hsv["s"] = s
    hsv["v"] = v

    ##print the resulting line charts
    fig, (ax_rgb, ax_hsv) = plt.subplots(2, figsize=(13, 8))
    fig.subplots_adjust(hspace=0.4)
    rgb.plot.line(x = "x", y = ["blue", "red", "green"], ax = ax_rgb, linewidth = 2)
    hsv.plot.line(x = "x", y = ["h", "s", "v"], ax = ax_hsv, linewidth = 2)
    ax_rgb.set_title("RGB-gradient", fontsize = 20)
    ax_hsv.set_title("HSV-gradient", fontsize = 20)
    ax_rgb.set_xlabel("position in colormap", fontsize = 12)
    ax_hsv.set_xlabel("position in colormap", fontsize = 12)
    ax_rgb.set_ylabel("RGB-value", fontsize = 12)
    ax_hsv.set_ylabel("HSV-value", fontsize = 12)
    ax_rgb.set_xlim([0,1])
    ax_hsv.set_xlim([0,1])
    ax_rgb.set_ylim([0,1.01])
    ax_hsv.set_ylim([0,1.01])
    ax_rgb.tick_params(axis='both', labelsize=12)
    ax_hsv.tick_params(axis='both', labelsize=12)
    ax_rgb.legend(fontsize = 12)
    ax_hsv.legend(fontsize = 12)

    #plt.savefig("visualizations\gradient.png", format = "png", dpi = 300)


##color differences between all colors

def col_diff(colorscale):

    color_diff = list()
    sim = False

    cmap = cm.get_cmap(colorscale, 1000)

    for i in range(0,1000,1):
        delta = list()
        for j in range(0,1000,1):
            p1 = i/1000
            p2 = j/1000
            color1_lab = rgb2lab([cmap(p1)[0], cmap(p1)[1], cmap(p1)[2]])
            color2_lab = rgb2lab([cmap(p2)[0], cmap(p2)[1], cmap(p2)[2]])
            color1 = LabColor(lab_l=color1_lab[0], lab_a=color1_lab[1], lab_b=color1_lab[2])
            color2 = LabColor(lab_l=color2_lab[0], lab_a=color2_lab[1], lab_b=color2_lab[2])
            diff = delta_e_cie2000(color1, color2)
            delta.append(diff)
            if(diff < 0.001):
                sim = True
        color_diff.append(delta)

    if(sim == True):
        print("Very similar colors included!")
    
    return color_diff


##color differences: start and end colors of exponent

def col_diff_se(min_exp, max_exp, colorscale):

    ## number of exponents
    nr_exp = max_exp - min_exp

    cmap = cm.get_cmap(colorscale, 1000)

    exp = list(range(1, nr_exp+1, 1))
    delta = list()

    ##compare two exponent-colors and create resulting dataframe
    for i in range(0,nr_exp,1):
        p1 = (i*int(1000/nr_exp))/1000
        p2 = ((i+1)*int(1000/nr_exp)-1)/1000
        color1_lab = rgb2lab([cmap(p1)[0], cmap(p1)[1], cmap(p1)[2]])
        color2_lab = rgb2lab([cmap(p2)[0], cmap(p2)[1], cmap(p2)[2]])
        color1 = LabColor(lab_l=color1_lab[0], lab_a=color1_lab[1], lab_b=color1_lab[2])
        color2 = LabColor(lab_l=color2_lab[0], lab_a=color2_lab[1], lab_b=color2_lab[2])
        delta.append(delta_e_cie2000(color1, color2))

    ##creating the scatterplot
    fig = plt.figure(figsize=(13, 8))

    plt.bar(
        x = exp,
        height = delta,
    )

    plt.title("Color-Differences (Distances between start- and end-colors of exponents)", fontsize = 20)
    plt.xlabel("Exponent", fontsize = 20)
    plt.ylabel('Delta-E', fontsize = 20)



##color differences: adjacent colors

def col_diff_adj(colorscale):
    
    ##get values of the cmap
    cmap = cm.get_cmap(colorscale, 1000)

    delta_e = pd.DataFrame()
    x = list()
    delta = list()
    ##compare two adjacent colors and create resulting dataframe
    for i in range(0,1000,1):
        j = i/1000
        k = (i+1)/1000
        x.append(j)
        color1_lab = rgb2lab([cmap(j)[0], cmap(j)[1], cmap(j)[2]])
        color2_lab = rgb2lab([cmap(k)[0], cmap(k)[1], cmap(k)[2]])
        color1 = LabColor(lab_l=color1_lab[0], lab_a=color1_lab[1], lab_b=color1_lab[2])
        color2 = LabColor(lab_l=color2_lab[0], lab_a=color2_lab[1], lab_b=color2_lab[2])
        delta.append(delta_e_cie2000(color1, color2))

    delta_e["x"] = x
    delta_e["delta"] = delta

    ##plot the results
    fig, axs = plt.subplots(figsize=(13, 8))
    delta_e.plot.line(x = "x", y = "delta", ax = axs, linewidth = 2)
    axs.set_title("Color-Differences (Distances between two adjacent colors)", fontsize = 20)
    axs.set_xlabel("position in colormap", fontsize = 15)
    axs.set_ylabel("Delta_E-value", fontsize = 15)
    axs.get_legend().remove()
    axs.tick_params(axis='both', labelsize=15)
    axs.set_xlim([0,1])
    axs.set_ylim([0,0.5])

    #plt.savefig("visualizations\deltaE_adj.png", format = "png", dpi = 300)


##color difference: background

def col_diff_back(colorscale):

    ##get values of the cmap
    cmap = cm.get_cmap(colorscale, 1000)

    delta_e = pd.DataFrame()
    x = list()
    delta = list()

    ##rgb values for color white
    white_rgb = [1, 1, 1]
    white_lab = rgb2lab([white_rgb[0], white_rgb[1], white_rgb[2]])
    white = LabColor(lab_l=white_lab[0], lab_a=white_lab[1], lab_b=white_lab[2])

    ##compare colors to the background color (i.e. white)
    for i in range(0,1001,1):
        j = i/1000
        x.append(j)
        color_lab = rgb2lab([cmap(j)[0], cmap(j)[1], cmap(j)[2]])
        color = LabColor(lab_l=color_lab[0], lab_a=color_lab[1], lab_b=color_lab[2])
        delta.append(delta_e_cie2000(color, white))

    delta_e["x"] = x
    delta_e["delta"] = delta

    ##plot the results
    fig, axs = plt.subplots(figsize=(13, 8))
    delta_e.plot.line(x = "x", y = "delta", ax = axs, linewidth = 2)
    axs.set_title("Color-Differences (Difference between colors and white background)", fontsize = 20)
    axs.set_xlabel("position in colormap", fontsize = 15)
    axs.set_ylabel("Delta_E-value", fontsize = 15)
    axs.get_legend().remove()
    axs.tick_params(axis='both', labelsize=15)
    axs.set_xlim([0,1])
    axs.set_ylim([0,0.5])

    #plt.savefig("visualizations\deltaE_back.png", format = "png", dpi = 300)


##color difference: each exponent

def col_diff_exp(min_exp, max_exp, colorscale):

    ## number of exponents
    nr_exp = max_exp - min_exp

    ##create plot matrix
    fig, axs = plt.subplots(nr_exp, nr_exp, figsize=(30, 15))

    cmap = cm.get_cmap(colorscale, 1000)

    x = np.linspace(0, 1, int(1000/nr_exp))

    ##compare two exponent-colors and create resulting dataframe
    for i in range(0,nr_exp,1):
        for j in range(0,nr_exp,1):
            y = list()
            for k in range(0,int(1000/nr_exp),1):
                p1 = (i*int(1000/nr_exp)+k)/1000
                p2 = (j*int(1000/nr_exp)+k)/1000
                color1_lab = rgb2lab([cmap(p1)[0], cmap(p1)[1], cmap(p1)[2]])
                color2_lab = rgb2lab([cmap(p2)[0], cmap(p2)[1], cmap(p2)[2]])
                color1 = LabColor(lab_l=color1_lab[0], lab_a=color1_lab[1], lab_b=color1_lab[2])
                color2 = LabColor(lab_l=color2_lab[0], lab_a=color2_lab[1], lab_b=color2_lab[2])
                y.append(delta_e_cie2000(color1, color2))
            axs[j, i].plot(x, y)
            axs[j, i].set_title('Exp ' + str(i+1) + ' vs. ' + str(j+1))
            axs[j, i].set_xlim([0,1])
            axs[j, i].set_ylim([-0.01,0.7])

    ## set x- / y-label for outer plots
    for ax in axs.flat:
        ax.set(xlabel='position', ylabel='Delta-E')

    ## Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    ##figure title
    fig.suptitle("Color Differences per Exponent", fontsize = 20)
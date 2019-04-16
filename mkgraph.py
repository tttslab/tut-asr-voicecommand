import matplotlib.pyplot as plt
import argparse

''' this is a simple script for graph making

usage:
python mkgraph.py -xl 'xlabel_name' -x x1 x2 ... xn -yl 'ylabel_name' -y y1 y2 ... yn -f 'save_file_name'
or 
python mkgraph.py --xlabel 'xlabel_name' --xvalue x1 x2 ... xn --ylabel 'ylabel_name' --yvalue y1 y2 ... yn --filename 'save_file_name'

argument order can be changed

example
python mkgraph.py -xl 'train(%)' -x 40 50 60 -yl 'eval accuracy(%)' -y 67.5 78.0 88.5 -f result.jpg

'''

parser = argparse.ArgumentParser()
parser.add_argument('-xl', '--xlabel', type=str, default=None, required=True)
parser.add_argument('-yl', '--ylabel', type=str, default=None, required=True)
parser.add_argument('-f', '--filename', type=str, default=None, required=True)
parser.add_argument('-x', '--xvalue', type=float, nargs='+', required=True)
parser.add_argument('-y', '--yvalue', type=float, nargs='+', required=True)
args = parser.parse_args()

print(len(args.xvalue))
if len(args.xvalue) != len(args.yvalue):
    print("invalid inputs: num_x must equal to num_y")
    sys.exit()

plt.plot(args.xvalue, args.yvalue, 'ro-')
plt.xlabel(args.xlabel)
plt.ylabel(args.ylabel)
plt.savefig(args.filename)

print('figure %s saved' % args.filename)

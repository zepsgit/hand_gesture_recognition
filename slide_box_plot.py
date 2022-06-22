import csv
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
fig, ax = plt.subplots(2,constrained_layout=True)
fig.supylabel('precision divergence')
fig.supxlabel('box location')
diffs=[]
for n in [1,2]:
    diff=[]
    with open('./dataset/slide_diff_{n}x{n}'.format(n=n), 'r') as fd:
        reader = csv.reader(fd)
        for row in reader:
            v=(round(float(row[0]),9))
            diff.append(0.766-v)
    diffs.append(diff)

for i in range(len(diffs)):
    x=[i for i in range(len(diffs[i]))]
    ax[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax[i].plot(x,diffs[i])
    ax[i].title.set_text('{n}x{n} box'.format(n=2**i))
plt.savefig('./img/slide_diff_updown.png')
plt.show()

# save box_1x1
# diff=[]
# with open('./dataset/slide_diff_1x1', 'r') as fd:
#     reader = csv.reader(fd)
#     for row in reader:
#         v=(round(float(row[0]),9))
#         diff.append(0.766-v)
# fig, ax = plt.subplots()
# x=[i for i in range(len(diff))]
# ax.plot(x, diff)

# ax.set(xlabel='box location', ylabel='precision difference',
#        title='1x1 box')
# # Change major ticks to show every 20.
# ax.xaxis.set_major_locator(MultipleLocator(10))
# ax.yaxis.set_major_locator(MultipleLocator(10))

# # Change minor ticks to show every 1. (10/10 = 1)
# ax.xaxis.set_minor_locator(AutoMinorLocator(10))
# ax.yaxis.set_minor_locator(AutoMinorLocator(10))

# ax.grid(which='major', color='#CCCCCC', linestyle='--')
# ax.grid(which='minor', color='#CCCCCC', linestyle=':')
# #fig.savefig("./img/diff_box_1x1_updown.png")
# plt.show()
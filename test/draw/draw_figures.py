import json
import matplotlib.pyplot as plt
import numpy as np

fontsize = (31, 29, 27)


'''Cumulative Distribution Function (CDF)'''
# with open('data/skipped_frames.txt') as f:
#     lost_frames = []
#     for line in f:
#         lost_frames.append(float(line[line.find('(') + 1:line.find('%') - 1]))
lost_frames = []
plt.rcParams["font.family"] = 'Times New Roman'

plt.hist(lost_frames, bins=len(lost_frames)*5, density=True, cumulative=True, color='tab:blue', histtype='step', lw=2)

plt.xticks(plt.xticks()[0], [f'{x:.0f}' for x in plt.xticks()[0]], fontsize=fontsize[2])
plt.yticks(plt.yticks()[0], [f'{x:.1f}' for x in plt.yticks()[0]], fontsize=fontsize[2])
plt.xlim(0, max(lost_frames))
plt.ylim(0, 1)
plt.xlabel('Frame Loss Rate (%)', fontsize=fontsize[0])
plt.ylabel('Cumulative Distribution', fontsize=fontsize[0])

plt.savefig('figs/fig4.png', dpi=300, bbox_inches='tight')
plt.savefig('figs/fig4.eps', bbox_inches='tight')
plt.clf()


'''Grouped Bar Chart w/o Error Bar'''
with open('data/bar_chart_data.json', 'r') as f:
    avg_cr, std_cr, avg_nocr, std_nocr = json.load(f)

plt.rcParams["font.family"] = 'Times New Roman'

bar_width = .4
x = np.arange(1, 13)
plt.bar(x, avg_cr, bar_width, align='center', color='tab:blue', label='CR')
plt.bar(x + bar_width, avg_nocr, bar_width, align='center', color='tab:orange', label='NoCR')

plt.xticks(x + bar_width / 2, [f'{i}' for i in range(1, 13)], fontsize=fontsize[2])
plt.yticks(fontsize=fontsize[2])
plt.xlabel('Activitiy', fontsize=fontsize[0])
plt.ylabel('No. Points', fontsize=fontsize[0])
plt.legend(fontsize=fontsize[2], loc='lower left')

plt.savefig('figs/fig5.png', dpi=300, bbox_inches='tight')
plt.savefig('figs/fig5.eps', bbox_inches='tight')
plt.clf()


'''w/ Error Bar'''
plt.bar(x, avg_cr, bar_width, yerr=std_cr, align='center', color='tab:blue', ecolor='black', capsize=3, label='CR')
plt.bar(x + bar_width, avg_nocr, bar_width, yerr=std_nocr, align='center', color='tab:orange', ecolor='black', capsize=3, label='NoCR')

plt.xticks(x + bar_width / 2, [f'{i}' for i in range(1, 13)], fontsize=fontsize[2])
plt.yticks(fontsize=fontsize[2])
plt.xlabel('Activitiy', fontsize=fontsize[0])
plt.ylabel('No. Points', fontsize=fontsize[0])
plt.legend(fontsize=fontsize[2], loc='lower left')

plt.savefig('figs/fig5_w_error_bar.png', dpi=300, bbox_inches='tight')
plt.savefig('figs/fig5_w_error_bar.eps', bbox_inches='tight')
plt.clf()


'''Line Graph (Average Velocity)'''
with open('data/CR_s20_a04_r02_mmw.json') as f:
    d1 = json.load(f)
    n_frames, n_points = d1.pop('n_frames'), d1.pop('n_points')
    for i in range(n_frames):
        if str(i) not in d1:
            d1[str(i)] = []
    for frame in d1.values():
        frame[:] = list(filter(lambda p: -.75 < p['x'] < .75 and -.5 < p['y'] < 1 and 1 < p['z'] < 2, frame))
with open('data/CR_s20_a07_r02_mmw.json') as f:
    d2 = json.load(f)
    n_frames, n_points = d2.pop('n_frames'), d2.pop('n_points')
    for i in range(n_frames):
        if str(i) not in d2:
            d2[str(i)] = []
    for frame in d2.values():
        frame[:] = list(filter(lambda p: -.75 < p['x'] < .75 and -.5 < p['y'] < 1 and 1 < p['z'] < 2, frame))
        
plt.rcParams["font.family"] = 'Times New Roman'

d1, d2 = sorted(d1.items(), key=lambda x: int(x[0])), sorted(d2.items(), key=lambda x: int(x[0]))
x, y = zip(*d1)
plt.plot(list(map(int, x)), list(map(lambda frame: sum(map(lambda point: point['velocity'], frame)) / len(frame) if len(frame) else 0, y)), color='tab:blue', ls='-', lw=1, marker='o', ms=8, markevery=65, label='a04')
x, y = zip(*d2)
plt.plot(list(map(int, x)), list(map(lambda frame: sum(map(lambda point: point['velocity'], frame)) / len(frame) if len(frame) else 0, y)), color='tab:orange', ls='--', dashes=(12, 4), lw=1, marker='s', ms=8, markevery=65, label='a07')

plt.xticks(fontsize=fontsize[2])
plt.yticks(fontsize=fontsize[2])
plt.xlabel('Frame', fontsize=fontsize[0])
plt.ylabel('Avg. Velocity (m/s)', fontsize=fontsize[0])
plt.legend(fontsize=fontsize[1], loc='best')

plt.savefig('figs/fig6.png', dpi=300, bbox_inches='tight')
plt.savefig('figs/fig6.eps', bbox_inches='tight')
plt.clf()


'''Line Graph (Average Bearing)'''
with open('data/CR_s09_a08_r02_mmw.json') as f:
    d1 = json.load(f)
    n_frames, n_points = d1.pop('n_frames'), d1.pop('n_points')
    for i in range(n_frames):
        if str(i) not in d1:
            d1[str(i)] = []
    for frame in d1.values():
        frame[:] = list(filter(lambda p: -.75 < p['x'] < .75 and -.5 < p['y'] < 1 and 1 < p['z'] < 2, frame))
with open('data/CR_s10_a08_r02_mmw.json') as f:
    d2 = json.load(f)
    n_frames, n_points = d2.pop('n_frames'), d2.pop('n_points')
    for i in range(n_frames):
        if str(i) not in d2:
            d2[str(i)] = []
    for frame in d2.values():
        frame[:] = list(filter(lambda p: -.75 < p['x'] < .75 and -.5 < p['y'] < 1 and 1 < p['z'] < 2, frame))
        
plt.rcParams["font.family"] = 'Times New Roman'

d1, d2 = sorted(d1.items(), key=lambda x: int(x[0])), sorted(d2.items(), key=lambda x: int(x[0]))
x, y = zip(*d1)
plt.plot(list(map(int, x)), list(map(lambda frame: sum(map(lambda point: point['bearing'], frame)) / len(frame) if len(frame) else 0, y)), color='tab:blue', ls='-', lw=1, marker='o', ms=8, markevery=65, label='Right')
x, y = zip(*d2)
plt.plot(list(map(int, x)), list(map(lambda frame: sum(map(lambda point: point['bearing'], frame)) / len(frame) if len(frame) else 0, y)), color='tab:orange', ls='--', dashes=(12, 4), lw=1, marker='s', ms=8, markevery=65, label='Left')

plt.xticks(fontsize=fontsize[2])
plt.yticks(fontsize=fontsize[2])
plt.xlabel('Frame', fontsize=fontsize[0])
plt.ylabel('Avg. Bearing (deg)', fontsize=fontsize[0])
plt.legend(fontsize=fontsize[1], loc='best')

plt.savefig('figs/fig7.png', dpi=300, bbox_inches='tight')
plt.savefig('figs/fig7.eps', bbox_inches='tight')
plt.clf()

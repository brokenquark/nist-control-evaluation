from audioop import reverse
import math
import domain
import utils
from typing import List
import statistics
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy, scipy
from typing import Tuple
import tabulate
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpmax, fpgrowth
from mlxtend.frequent_patterns import association_rules
from prefixspan import PrefixSpan
import functools
from itertools import count
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import networkx as nx
from scipy import stats

controls : List['domain.Control'] = []
tactics : List['domain.Tactic'] = []
techniques : List['domain.Technique'] = []
procedures  : List['domain.Procedure'] = []
groups  : List['domain.Group'] = []
softwares  : List['domain.Software'] = []
cocTTPs : List[List['domain.Technique']] = []

utils.buildDataSchema(controls, tactics, techniques, procedures, groups, softwares, cocTTPs)


def countTechniqueAndControl(controls: List['domain.Control'], techniques: List['domain.Technique']) -> None:
    print('***')
    print('Descriptive statistics of the control and technique mapping')
    print(f'Number of mitigating controls: {len( [x for x in controls if x.tec > 0] )}')
    print(f'Total number of controls: {len( controls )}')
    
    print(f'Number of mitigated techniques: {len( [x for x in techniques if x.scCount > 0] )}')
    print(f'Total number of techniques: {len( techniques )}')
    
    print(f'Number of singleton controls : {len( [x for x in controls if x.tec == 1] )}')
    print(f'Total number of singleton technique: {len( [x for x in techniques if x.scCount == 1] )}')
    
    tecValues = [x.tec for x in controls if x.tec > 0]
    scCountValues = [x.scCount for x in techniques if x.scCount > 0]
    
    print('techniques per control')
    print(f'Mean: {statistics.mean(tecValues)}')
    print(f'Median: {statistics.median(tecValues)}')
    print(f'Dev: {statistics.stdev(tecValues)}')
    print(f'Q1: {statistics.quantiles(tecValues, n=4)[0]}')
    print(f'Q3: {statistics.quantiles(tecValues, n=4)[2]}')
    print(f'Min: {min(tecValues)}')
    print(f'Max: {max(tecValues)}')
    
    print('control per techniques')
    print(f'Mean: {statistics.mean(scCountValues)}')
    print(f'Median: {statistics.median(scCountValues)}')
    print(f'Dev: {statistics.stdev(scCountValues)}')
    print(f'Q1: {statistics.quantiles(scCountValues, n=4)[0]}')
    print(f'Q3: {statistics.quantiles(scCountValues, n=4)[2]}')
    print(f'Min: {min(scCountValues)}')
    print(f'Max: {max(scCountValues)}')
    print('***\n')
    
    print('top controls')
    controls.sort(key=lambda c:c.tec, reverse=True)
    topTenControls = controls[0:10]
    utils.printTopN(controls, 10, 'tec', 'item')
    
    print('top techniques having the most mitigating techniques')
    techniques.sort(key=lambda t:len(t.controls), reverse=True)
    topTenTechniques = techniques[0:10]
    utils.printTopN(techniques, 10, 'tactics', 'list')
    utils.printTopN(techniques, 10, 'scCount', 'item')
    
    # for te in topTenTechniques:
    #     for ct in topTenControls:
    #         if ct in te.controls:
    #             print(f'{te.name} - {ct.name}')
    
    # print('unmitigated techniques')
    # umTechniques = [x for x in techniques if x.scCount == 0]
    # utils.printTopN(umTechniques, 35, 'tactics', 'list')
    
    # print('unmitigated techniques')
    # techniques.sort(key = lambda v : v.prob, reverse=True)
    # idx = 1
    # for te in techniques:
    #     if len(te.controls) == 0:
    #         print(f'{idx} {te.title}@{te.tactics[0].title} ==> {format(te.prob*100, ".2f")}')
    #     idx += 1
        
    return


def investigateControlRedundancy(controls):
    mcontrols = [x for x in controls if len(x.ttps) > 0]
    mcontrols.sort(key=lambda c : c.red, reverse=True)

    print('controls with most redundancy')
    for c in mcontrols[0:5]:
        print(f'{c.name}: {c.title} | {len(c.ttps)} | {c.red}')

    mcontrols.sort(key=lambda c : c.red, reverse=False)
    print('controls with least redundancy')
    for c in mcontrols[0:5]:
        print(f'{c.name}: {c.title} | {len(c.ttps)} | {c.red}')

    data = {
        'control': [x.name for x in mcontrols], 
        'TEC': [len(x.ttps) for x in mcontrols],
        'CR': [x.red for x in mcontrols],
        'prob': [x.prob for x in mcontrols]
    }

    df = pd.DataFrame(data)

    # print(df)

    sns.scatterplot(data=df, x="TEC", y="CR")
    # sns.scatterplot(data=df, x="prob", y="red")
    
    minCR = min(data['CR'])
    maxCR = max(data['CR'])
    minTEC = min(data['TEC'])
    maxTEC = max(data['TEC'])
    
    xvalues = [(minTEC+maxTEC)/2, (minTEC+maxTEC)/2]
    yvalues = [0, maxCR]
    plt.plot(xvalues, yvalues, 'bo', linestyle="--")
    
    xvalues = [0, maxTEC]
    yvalues = [(maxCR-minCR)/2+minCR, (maxCR-minCR)/2+minCR]
    plt.plot(xvalues, yvalues, linestyle="--")
    
    plt.xlim([minTEC-1, maxTEC+1])
    plt.ylim([minCR-1, maxCR+1])
    
    plt.text(80, 20, 'QT1: high TEC, high CR', fontsize=10)
    plt.text(20, 7.5, 'QT3: low TEC, low CR', fontsize=10)
    plt.text(80, 10, 'QT4: high TEC, low CR', fontsize=10)
    plt.text(20, 25, 'QT2: low TEC, high CR', fontsize=10)
    
    
    
    plt.show()

    idx = 0
    counts = [0,0,0,0]
    mcontrols.sort(key=lambda c : len(c.ttps), reverse=True)
    print('controls that have low redundancy but high coverage')
    for c in mcontrols:
        if c.red < 16.625 and len(c.ttps) < 60:
            # print(f'{c.name} | {len(c.ttps)} | {idx}')
            counts[0] += 1
        if c.red < 16.625 and len(c.ttps) > 60:
            print(f'{c.name}: {c.title} | {len(c.ttps)} | {idx}')
            counts[1] += 1
        if c.red > 16.625 and len(c.ttps) < 60:
            # print(f'{c.name} | {len(c.ttps)} | {idx}')
            counts[2] += 1
        if c.red > 16.625 and len(c.ttps) > 60:
            # print(f'{c.name} | {len(c.ttps)} | {idx}')
            counts[3] += 1
            
        idx += 1
    
    print(f'{counts}')
    
    print(f'{statistics.mean([x.red for x in mcontrols])} {statistics.median([x.red for x in mcontrols])}')
    
    ac8 = [x for x in mcontrols if x.name == 'AC-8'][0]
    for te in ac8.ttps:
        print(f'{te.name}: {te.title}')
    
    return


def findTopControlsForEachTactic(controls, tactics):
    mcontrols = [c for c in controls if len(c.ttps) > 0]
    tactics.sort(key=lambda v: v.name)

    idx = 0
    for ta in tactics:
        # c = max(mcontrols, key=lambda c : c.tac[idx])
        mcontrols.sort(key=lambda c : c.tac[idx], reverse=True)
        c1 = mcontrols[0]
        c2 = mcontrols[1]
        c3 = mcontrols[2]
        print(f'{ta.title} ==> {c1.name} {c2.name} {c3.name} {format(c1.tac[idx], ".3f")} {format(c2.tac[idx], ".3f")} {format(c3.tac[idx], ".3f")}')  
        idx += 1  
    return

def findTechniquesWithHighestAndLowestMitigations(techniques):
    mtechniques = [x for x in techniques if len(x.controls) > 0]
    mtechniques.sort(key = lambda v : len(v.controls), reverse=True)
    for t in mtechniques[0:5]:
        print(f'{t.name} | {t.title} | {t.tactics[0].title} | {len(t.controls)}')

    print('...')
    mtechniques.sort(key = lambda v : len(v.controls), reverse=False)
    for t in mtechniques[0:10]:
        print(f'{t.name} | {t.title} | {t.tactics[0].title} | {len(t.controls)}')
    
    return

def findTopControlsMTAC(controls, tactics):
    mcontrols = [x for x in controls if len(x.ttps) > 0]
    mcontrols.sort(key = lambda v : v.mtac, reverse=True)
    tactics.sort(key=lambda v: v.name)
    
    data = {}
    data['control'] = [f'{x.name}: {x.title}' for x in mcontrols]
    
    for c in mcontrols[0:10]:
        print(f'{c.name}: {c.title} | {format(c.mtac, ".2f")} | {c.tec}')
        idx = 0
        for ta in tactics:
            data[f'{ta.name}'] = [x.tac[idx] for x in mcontrols]
            # print(f'{ta.name}@{ta.title}: {format(c.tac[idx], ".2f")}')
            idx += 1
    
    data['MTAC'] = [x.mtac for x in mcontrols]
    df = pd.DataFrame(data)
    
    df['TA0001'] = df['TA0001'].round(decimals = 2)
    df['TA0002'] = df['TA0002'].round(decimals = 2)
    df['TA0003'] = df['TA0003'].round(decimals = 2)
    df['TA0004'] = df['TA0004'].round(decimals = 2)
    df['TA0005'] = df['TA0005'].round(decimals = 2)
    df['TA0006'] = df['TA0006'].round(decimals = 2)
    df['TA0007'] = df['TA0007'].round(decimals = 2)
    df['TA0008'] = df['TA0008'].round(decimals = 2)
    df['TA0009'] = df['TA0009'].round(decimals = 2)
    df['TA0010'] = df['TA0010'].round(decimals = 2)
    df['TA0011'] = df['TA0011'].round(decimals = 2)
    df['TA0040'] = df['TA0040'].round(decimals = 2)
    df['TA0042'] = df['TA0042'].round(decimals = 2)
    df['TA0043'] = df['TA0043'].round(decimals = 2)
    df['MTAC'] = df['MTAC'].round(decimals = 2)
    
    
    
    
    print(tabulate.tabulate(df, headers='keys', tablefmt='psql', showindex=False))
    
    print(df.describe())
    
    return

def computeAdvCov(controls):
    mcontrols = [c for c in controls if len(c.ttps) > 0]

    for control in mcontrols:
        count = 0
        for adv in cocTTPs:
            if not set(control.ttps).isdisjoint(set(adv)):
                count += 1
        control.advCov = count/len(cocTTPs)
        # print(f'{control.name}: {control.advCov}')

    mcontrols.sort(key = lambda v : v.advCov, reverse=True)

    print('top controls in adversary coverage')
    for control in mcontrols[0:10]:
        print(f'{control.name}: {control.title} & {format(control.advCov, ".2f")}')
    
    # print('bottom controls in adversary coverage')
    # for control in mcontrols[90:]:
    #     print(f'{control.name}: {control.advCov}')
    
    print(f'mean and median of AC')
    print(f'{statistics.mean([x.advCov for x in mcontrols])}--{statistics.median([x.advCov for x in mcontrols])}')
    
    return

def computeAdvTecCov(controls):
    mcontrols = [x for x in controls if len(x.ttps) > 0]

    for control in mcontrols:
            count = 0
            for adv in cocTTPs:
                count += len( list( set(control.ttps).intersection(set(adv)) ) ) / len(adv)
            control.advTecCov = count/len(cocTTPs)
            # print(f'{control.name}: {control.advTecCov}')

    mcontrols.sort(key = lambda v : v.advTecCov, reverse=True)

    print('top controls in adversary technique coverage')
    for control in mcontrols[0:10]:
        print(f'{control.name}: {format(control.advTecCov, ".2f")}')
    
    print(f'mean and median of ATC')
    print(f'{statistics.mean([x.advTecCov for x in mcontrols])}--{statistics.median([x.advTecCov for x in mcontrols])}')
    
    data = {
        'control': [x.name for x in mcontrols], 
        'AC': [x.advCov for x in mcontrols],
        'ATC': [x.advTecCov for x in mcontrols],
    }

    df = pd.DataFrame(data)

    # print(df)

    # sns.scatterplot(data=df, x="AC", y="ATC")
    # plt.show()
    return




    mcontrols = [c for c in controls if c.tec > 0]

    for control in mcontrols:
        sum = 0
        for te in control.ttps:
            sum += te.support * te.severity
        # control.risk = sum/len(control.ttps)
        control.risk = sum

    mcontrols.sort(key = lambda v : v.risk, reverse = True)

    print(f'*** Risk ***')
    for control in mcontrols:
        if control.tec > 2: print(f'{control.name}: {control.tec} {control.risk}')

    return

def findClosestCentroids(df, centroids):
    xlist = df['pc1'].tolist()
    ylist = df['pc2'].tolist()
    
    output = []
    
    for item in centroids:
        x = item[0]
        y = item[1]
        
        distance = 10000
        closestIndex = None
        for index in range(0, len(xlist)):
            newDistance = (x-xlist[index])*(x-xlist[index]) + (y-ylist[index])*(y-ylist[index])
            if newDistance < distance:
                distance = newDistance
                closestIndex = index
        output.append(closestIndex)
    
    return output

def cluster(controls : List['domain.Control'], techniques, tactics, cocTTPs) -> None:
    mcontrols : List['domain.Control'] = [x for x in controls if x.tec > 0]
    names = [x.name for x in mcontrols]
    tecList = [x.tec for x in mcontrols]
    tacList = [x.mtac for x in mcontrols]
    redList = [x.red for x in mcontrols]
    advCovList = [x.advCov for x in mcontrols]
    advTecCovList = [x.advTecCov for x in mcontrols]
    # advTacCovList = [statistics.median(x.advTacCov) for x in mcontrols]
    riskList = [x.risk for x in mcontrols]
    
    data = {
        'id' : names,
        'TEC': utils.normalizeList(tecList),
        'MTAC': utils.normalizeList(tacList),
        'CR': utils.normalizeList(redList),
        'AC': utils.normalizeList(advCovList),
        'ATC': utils.normalizeList(advTecCovList),
        # 'ATaC': utils.normalizeList(advTacCovList),
        'CMR': utils.normalizeList(riskList)
    }
    
    df = pd.DataFrame(data)
    # print(df)
    
    x = df.iloc[:, 1:7]
    pca_2 = PCA(n_components=2)
    pca_2_result = pca_2.fit_transform(x)
    # print(f'explanation ratio: {pca_2.explained_variance_ratio_}')
    df_pca  = pd.DataFrame(pca_2_result, columns=['pc1', 'pc2'])
    
    
    # investigate wcss
    
    # wcss = []
    # for i in range(1, 5):
    #     kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    #     kmeans.fit(df_pca)
    #     wcss.append(kmeans.inertia_)
    
    # plt.close()
    # plt.plot(range(1, 5), wcss)
    # plt.title('The Elbow Point Graph')
    # plt.xlabel('Number of clusters')
    # plt.ylabel('WCSS')
    # plt.show()
    
    
    kmeans = KMeans(2)
    kmeans.fit(df_pca)
    y_clusters = kmeans.fit_predict(df_pca)
    centroids = kmeans.cluster_centers_
    centroidIndex = findClosestCentroids(df_pca, centroids)
    df_pca['cluster'] = 'none'
    df['cluster'] = 'none'

    for index in range(0, len(y_clusters)):
        # if y_clusters[index] == 2: 
        #     df_pca.at[index, 'cluster'] = 'cluster-A'
        #     df.at[index, 'cluster'] = 'cluster-A'
        if y_clusters[index] == 0: 
            df_pca.at[index, 'cluster'] = 'Cluster-B'
            df.at[index, 'cluster'] = 'Cluster-B'
        # if y_clusters[index] == 3: 
        #     df_pca.at[index, 'cluster'] = 'cluster-C'
        #     df.at[index, 'cluster'] = 'cluster-C'
        if y_clusters[index] == 1: 
            df_pca.at[index, 'cluster'] = 'Cluster-A'
            df.at[index, 'cluster'] = 'Cluster-A'
    
    # print(df_pca.groupby(['cluster']).count())
    # print(df.groupby(['cluster']).count())
    dfg = df.groupby(['cluster'])
    print(dfg.count())
    df_pca.sort_values(by=['cluster'], ascending=True)
    # sns.scatterplot(data=df_pca, x="pc1", y="pc2", palette='bright', hue='cluster', style='cluster', marker=True)
    # plt.show()
    
    dfm = pd.melt(df, id_vars=['id', 'cluster'], value_vars=['TEC', 'MTAC', 'CR', 'AC', 'ATC', 'CMR'])
    dfm = dfm.rename(columns={'variable': 'metrics'})
    sns.boxplot(data=dfm, x='cluster', y='value', hue='metrics', palette='Greys')
    # sns.boxplot(data=dfm, x='variable', y='value', hue='cluster', palette='Greys')
    
    
    for name, group in dfg:
        ttps = []
        cc = []
        for row in group.itertuples():
            control = [x for x in mcontrols if x.name == row.id][0]
            cc.append(control)
            ttps.extend(control.ttps)
        ttps = list(set(ttps))
        print(f'{name} total techniques mitigated: {len(ttps)}')
        
        ccTAC = []
        for ta in tactics:
            ccTAC.append( len(list(set(ttps).intersection(set(ta.techniques)))) / len(ta.techniques) )
        
        print(ccTAC)
        print(f'MTAC: {statistics.mean(ccTAC)}')
        
        ccadvcount = 0
        for adv in cocTTPs:
            if set(ttps).isdisjoint(set(adv)) == False:
                ccadvcount += 1
        
        print(f'adv cov: {ccadvcount/len(cocTTPs)}')
        
        ccadvpc = []
        for adv in cocTTPs:
            pct =  len(list(set(ttps).intersection(set(adv)))) / len(adv)
            ccadvpc.append(pct)
        
        print(f'adv tec cov: {statistics.mean(ccadvpc)}')
        
        cc.sort(key = lambda c: c.cat)
        for c in cc:
            print(f'{c.name}: {c.title} & {c.cat} \\\\')
        print('..')
    
    plt.show()
    
    return

def computeControlRiskAltNonGraph(cocTTPs, controls, techniques):
    for te1 in techniques:
        timesTe1 = 0
                
        for adv in cocTTPs:
            if te1 in adv: timesTe1 += 1
        
        timesTe1AndTe2 = [0] * 188
        idx = 0
        for te2 in techniques:
            for adv in cocTTPs:
                if te1 != te2 and te1 in adv and te2 in adv:
                    timesTe1AndTe2[idx] += 1
            idx += 1
        
        te1.ec = len( [x for x in timesTe1AndTe2 if x > 30] ) / 187
        
        if te1.ec > 0:
            count = []
            for item in timesTe1AndTe2:
                if item > 30 : count.append( (1- (item / timesTe1)) )
            
            te1.hf = statistics.mean(count)
            te1.severity = math.exp(-te1.hf/8) * te1.ec
        else: 
            te1.hf = 0
            te1.severity = 0
    
    for c in controls:
        if c.tec > 0: 
            c.risk = sum([ x.severity for x in c.ttps])
            
    return


def computeControlRiskAltNonGraphAlt(cocTTPs, controls, techniques, tactics):
    for te1 in techniques:
        timesTe1 = 0
                
        for adv in cocTTPs:
            if te1 in adv: timesTe1 += 1
        
        # timesTe1AndTe2 = [0] * 188
        # idx = 0
        # for te2 in techniques:
        #     for adv in cocTTPs:
        #         if te1 != te2 and te1 in adv and te2 in adv:
        #             timesTe1AndTe2[idx] += 1
        #     idx += 1
        
        # te1.ec = len( [x for x in timesTe1AndTe2 if x > 30] ) / 187
        
        timesTe1AndTa = [0] * 14
        idx = 0
        for ta in tactics:
            for adv in cocTTPs:
                ttpsInThisTa = list(set(ta.techniques).intersection(set(adv)))
                if (te1 in adv) and len(ttpsInThisTa) > 0:
                    if len(ttpsInThisTa) == 1:
                        if ttpsInThisTa[0] != te1:
                            timesTe1AndTa[idx] += 1
                        else:
                            pass
                    else:
                        timesTe1AndTa[idx] += 1
            idx += 1
        
        te1.ec = len( [x for x in timesTe1AndTa if x > 0] ) / 14
        
        if te1.ec > 0:
            count = []
            for item in timesTe1AndTa:
                if item > 0 : count.append( (1- (item / timesTe1)) )
            
            te1.hf = statistics.mean(count)
            te1.severity = math.exp(-te1.hf/8) * te1.ec
            te1.risk = te1.support * te1.severity
        else: 
            te1.hf = 0
            te1.severity = 0
            te1.risk = te1.support * te1.severity
    
    for c in controls:
        if c.tec > 0: 
            c.risk = sum([ x.risk for x in c.ttps])
    
    techniques.sort(key = lambda v : v.risk, reverse=True)
    
    for te in techniques[0:10]:
        print(f'{te.name}:{te.title} ==> {te.risk}')
    
    controls.sort(key = lambda c : c.risk, reverse=True)
    print('top ten risky controls')
    for c in controls[0:10]:
        print(f'{c.name}: {c.title} & {format(c.risk, ".2f")} & {format(c.tec, ".2f")} \\\\')
    cmrlist = [x.risk for x in controls if x.tec > 0]
    print(f'mean: {statistics.mean(cmrlist)} {statistics.quantiles(cmrlist, n=4)}')
         
    return



def computeControlRiskAlt(cocTTPs, controls, techniques):
    transactions = []

    for cases in cocTTPs:
        transaction = []
        transaction.extend( [x.name for x in cases] )
        transactions.append(transaction)

    # print(transactions)
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    # print(df.head())

    frequent_itemsets = fpgrowth(df, min_support=0.05, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.25)

    rules['alen'] = rules['antecedents'].apply(lambda x : len(x)) 
    rules['clen'] = rules['consequents'].apply(lambda x : len(x)) 
    dfq = rules.query("alen + clen == 2")[['antecedents', 'consequents', 'support', 'confidence']]

    mcontrols = [x for x in controls if x.tec > 0]

    for control in mcontrols:
        count = 0
        for row in dfq.itertuples():
            te1 = next((x for x in techniques if x.name == list(row.antecedents)[0]), None)
            te2 = next((x for x in techniques if x.name == list(row.consequents)[0]), None)
            
            if control in te1.controls and control in te2.controls:
                count += 1
        control.rulesCov = count / len(dfq)

    mcontrols.sort(key = lambda c : c.rulesCov, reverse=True)

    # print('...')
    # for c in mcontrols[0:10]:
    #     print(f'{c.name}: {c.rulesCov}')

    # building graph
    ttpsTuples = []
    for row in dfq.itertuples():
        ttpsTuples.append([( list(row.antecedents)[0], list(row.consequents)[0]  ), row.confidence, row.support])
    techniqueNames = [x.name for x in techniques]
    edges = []
    for item in ttpsTuples:
        edges.append([(item[0][0], item[0][1]), item[1], item[2]])
    # cocDiGraph = nx.DiGraph()
    cocDiGraph = nx.Graph()
    for item in edges:
        if item[0][0] not in list(cocDiGraph.nodes):
            cocDiGraph.add_node(item[0][0])
        if item[0][1] not in list(cocDiGraph.nodes):
            cocDiGraph.add_node(item[0][1])
        
        cocDiGraph.add_edge(item[0][0], item[0][1], weight = item[1], count = len(cocTTPs) * item[2], distance = 1 - item[1])

    for node in cocDiGraph.nodes():
        cocDiGraph.nodes[node]['tactic'] = next( (x.tactics[0].name for x in techniques if x.name == node) )
        te = next( (x for x in techniques if x.name == node) )
        cocDiGraph.nodes[node]['frequency'] = len([x for x in cocTTPs if te in x])
        cocDiGraph.nodes[node]['title'] = f'{cocDiGraph.nodes[node]["tactic"]}:{node}'

    ecDict = nx.eigenvector_centrality(cocDiGraph)
    ecDict = nx.degree_centrality(cocDiGraph)

    for te in techniques:
        for node in cocDiGraph.nodes():
            if te.name == node:
                # print(f'{list(cocDiGraph.pred[node].keys())}')
                neighbors = list(cocDiGraph.adj[node].keys())
                # print(f'{cocDiGraph.edges[te.name, neighbors[0]]}')
                count = 0
                for nd in neighbors:
                    count += (1 - cocDiGraph.edges[te.name, nd]['weight'])
                te.hf = count / len(neighbors)
                te.ec = ecDict[f'{te.name}']

    for control in mcontrols:
        control.risk = sum([ math.exp(-x.hf/8) * x.ec * x.prob for x in control.ttps])
    
    mcontrols.sort(key = lambda v : v.risk, reverse = True)
    
    print(f'*** alternate risk ***')
    for control in mcontrols:
        if control.tec > 2: print(f'{control.name}: {control.tec} {control.risk}')
    
    techniques.sort(key = lambda x : math.exp(-x.hf/8) * x.ec * x.prob, reverse = True )
    techniques.sort(key = lambda x : x.support * x.severity, reverse = True )
    
    for te in techniques:
        print(f'{te.title} {te.support * te.severity}')
    
    return

def investigateUnmitigatedTechniques(techniques, cocTTPs):
    print(f'unmitigated adversaries')
    count = 0
    unmitigatedPct = []
    for adv in cocTTPs:
        for te in adv:
            if te.scCount == 0:
                count += 1
                break
    print(f'unmitigated adv count: {count}')

    print(f'unmitigated pct adversaries')
    unmitigatedPct = []
    for adv in cocTTPs:
        count = 0
        for te in adv:
            if te.scCount == 0:
                count += 1
                te.timesUnmitigated += 1
        unmitigatedPct.append(count/len(adv))

    print(f'min, mean, max: {min(unmitigatedPct)} | {statistics.mean(unmitigatedPct)} | {max(unmitigatedPct)}')
    print(f'Q1 Q2 Q3: {statistics.quantiles(unmitigatedPct, n=4)}')

    techniques.sort(key = lambda v : v.timesUnmitigated, reverse=True)
    for te in techniques[0:10]:
        print(f'{te.name}:{te.title} - {te.tactics[0].name}:{te.tactics[0].title} & {format(te.timesUnmitigated/len(cocTTPs), ".2f")} \\\\')
    return

def countControlTactics(controls : List['domain.Control'], tactics : List['domain.Tactic']) -> None :
    print('***')
    print(f'tactic | # members | # mitigated | # controls')
    for tactic in tactics:
        print(f'{tactic.name} | {len(tactic.techniques)} | {len( [x for x in tactic.techniques if x.scCount > 0] ) } | {len(tactic.controls)}')
        print(f'{tactic.name} # of controls: ')
    print('***\n')
    return  



countTechniqueAndControl(controls, techniques)
findTechniquesWithHighestAndLowestMitigations(techniques)
findTopControlsForEachTactic(controls, tactics)
countControlTactics(controls, tactics)
findTopControlsMTAC(controls, tactics)
investigateControlRedundancy(controls)
computeAdvCov(controls)
computeAdvTecCov(controls)
computeControlRiskAltNonGraphAlt(cocTTPs, controls, techniques, tactics)
investigateUnmitigatedTechniques(techniques, cocTTPs)
cluster(controls, techniques, tactics, cocTTPs)


mcontrols = [x for x in controls if x.tec > 0]
mcontrols.sort(key=lambda v: v.tec, reverse=True)

for control in mcontrols:
    print(f'{control.name} & {control.tec} & {format(control.mtac, ".2f")} & {format(control.red, ".2f")} & {format(control.advCov, ".2f")} & {format(control.advTecCov, ".2f")} & {format(control.risk, ".2f")} \\\\')
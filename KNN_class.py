#B711093 성의현
import numpy as np
import math

#KNN을 위한 class선언
class KNN :
    #생성자
    def __init__(self, data, target, target_names, k) :
        self.data = data
        self.target = target
        self.target_names = target_names
        self.K = k
    
    #데이터간의 거리 측정 함수(유클리디안 거리 측정 공식 이용)
    def dist(self, data_1, data_2) :
        sum = 0
        for i in range(0, len(data_1)) :
            tmp = data_2[i] - data_1[i]
            tmp = tmp * tmp
            sum += tmp
        return math.sqrt(sum)
    
    #k개의 가장 가까운 data들을 [데이터간의 거리, 데이터의 target 숫자]의 형식으로된 배열로 return
    def k_neighbor(self, data) :
        result = []
        #모든 데이터에 대해 거리와 target저장
        for i in range(0, len(self.data)) :
            data_set = []
            data_set.append(self.dist(data, self.data[i]))
            data_set.append(self.target[i])
            result.append(data_set)
        result.sort(key=lambda x:x[0])#각 데이터와의 거리를 정렬
        return result[0:self.K] #거리가 가장 작은 K개의 데이터 return
    
    #단순하게 가장 많은 데이터를 골라주는 함수
    def m_vote(self, set) :
        #target의 종류의 수에 맞게 array생성. [target의 index, 해당 index에 해당하는 데이터의 개수]를 요소로 하는 array
        vote_count = []
        for i in range(0, len(self.target_names)) :
            tmp = [i, 0]
            vote_count.append(tmp)
        
        #count하는 과정
        for i in range(0, self.K) :
            vote_count[set[i][1]][1] +=  1 #set[i][1]은 i + 1번째로 가까운 데이터의 target종류이다
        
        #가장 많은 종의 iris를 target에 맞게 return
        vote_count.sort(key=lambda x:x[1], reverse=True)
        return self.target_names[vote_count[0][0]]
    
    #거리를 확인해서 가까운 데이터일 수록 반영이 많이 되는 vote함수
    def w_m_vote(self, set) :
        #target의 종류의 수에 맞게 array생성 :  [target의 index, 해당 index에 해당하는 데이터의 개수]를 요소로 하는 array
        vote_count = []
        for i in range(0, len(self.target_names)) :
            tmp = [i, 0]
            vote_count.append(tmp)

        #count하는 과정으로, 가장 가까운 i일 수록 큰 투표수를 가진다.
        for i in range(0, self.K) :
            vote_count[set[i][1]][1] = vote_count[set[i][1]][1] + (self.K - i) #set[i][1]은 i + 1번째로 가까운 데이터의 target종류이다
        
        #가장 많은 종의 iris를 target에 맞게 return
        vote_count.sort(key=lambda x:x[1], reverse=True)
        return self.target_names[vote_count[0][0]]

    
    #들어온 데이터에 대해 판단한 target을 출력하는 함수, type지정을 통해 다수결방식인지 weighted다수결방식인지 정할 수 있다.
    def test(self, data, type) : 
        neighbor_set = np.empty((0,2))
        neighbor_set = self.k_neighbor(data) #neighbor_set은 [데이터간의 거리, 데이터의 target]으로 저장

        if(type == 'vote') : return self.m_vote(neighbor_set) #다수결방식
        else : return self.w_m_vote(neighbor_set) #weighted 다수결방식
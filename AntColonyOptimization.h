#ifndef ANTCOLONYOPTIMIZATION_H_INCLUDED
#define ANTCOLONYOPTIMIZATION_H_INCLUDED

#include <iostream>
#include<vector>
#include<Eigen/Dense>
#include<math.h>
#define MAX_VALUE 9999

using namespace std;

////////////////////////��������////////////////////////////
//�������飬������±��ʾ����ı�ţ������ֵ��ʾ����ĳ��ȡ����磺tasks[0]=10��ʾ��һ����������񳤶���10.
int *tasks;
//�����������
const int taskNum=100;

//����ڵ�����顣������±��ʾ����ڵ�ı�ţ�����ֵ��ʾ�ڵ�Ĵ����ٶ�
int *nodes;
//����ڵ������
const int nodeNum=10;

//��Ⱥ�����Ĵ���
int iteratorNum;
//ÿ�ε��������ϵ�������ÿֻ���϶���һ����������ߣ�ÿ�ε����е�ÿһֻ���϶���Ҫ�����������ķ��䣬��Ҳ����һ�����н⡣
int antNum;

//������ʱ���������һ����ά���󡣱��磺timeMatrix[i][j]�ͱ�ʾ��i������������j���ڵ�����Ĵ���ʱ�䡣
//timeMatrix[i][j]=task[i]/nodes[j]
MatrixXd timeMatrix;
//pheromoneMatrix����Ϣ�ؾ��� pheromoneMatrix[i][j]=0.5�ͱ�ʾ����i������ڵ�j����·���ϵ���Ϣ��Ũ��Ϊ0.5
MatrixXd pheromoneMatrix;
//maxPheromoneMatrix��pheromoneMatrix�����ÿһ���������Ϣ�ص��±�
std::vector<int> maxPheromoneMatrix[taskNum];
//criticalPointMatrix ��һ�ε����У��������������Ե����ϵ��ٽ��š�
std::vector<int> criticalPointMatrix[taskNum];

//p��ÿ���һ�ε�������Ϣ��˥���ı�����
int p=0.5;
//q������ÿ�ξ���һ��·������Ϣ�����ӵı�����
int q=2;
//������ʱ��Ľ���ܼ�
std::vector<std::vector<int>> resultData;

void initialTasksAndNodes() {
    mt19937 rng;
    //��ʼ�����񼯺�
    tasks=new int[taskNum];
    for(int aa=0; aa<taskNum; aa++) {
        std::uniform_int_distribution<int> dis1(10,100);
        tasks[aa]=dis1(rng);
    }
    nodes=new int[nodeNum];
    for(int i=0; i<nodeNum; i++) {
        std::uniform_int_distribution<int> dis1(10,100);
        nodes[i]=dis1(rng);
    }
}
////////////////////////��������////////////////////////////
void initTimeMatrix(int tasks[],int nodes[]);
void initPheromoneMatrix(int taskNum,int nodeNum);
void acaSearch(int iteratorNum,int antNum);
int assignOneTask(int antCount,int taskCount,int* nodes,MatrixXd pheromoneMatrix);
std::vector<int> calTime_oneIt(std::vector<MatrixXd> pathMatrix_allAnt);
void updatePheromoneMatrix(std::vector<MatrixXd> pathMatrix_allAnt,MatrixXd pheromoneMatrix,std::vector<int> timeArray_oneIt);
/***
��Ⱥ�㷨
*/
void aca() {
    //��ʼ������ִ��ʱ�����
    initTimeMatrix(tasks,nodes);
    //��ʼ����Ϣ�ؾ���
    initPheromoneMatrix(taskNum,nodeNum);
    //��������
    acaSearch(iteratorNum,antNum);
}

/**
 * ��ʼ��������ʱ�����
 * @param tasks ����(����)�б�
 * @param nodes �ڵ�(�����ٶ�)�б�
 */
void initTimeMatrix(int tasks[],int nodes[]) {
    for(int i=0; i<taskNum; i++) {
        for(int j=0; j<nodeNum; j++) {
            timeMatrix(i,j)=tasks[i]/nodes[j];
        }
    }
}

/**
 * ��ʼ����Ϣ�ؾ���(ȫΪ1)
 * @param taskNum ��������
 * @param nodeNum �ڵ�����
    ȫ����ʼ��Ϊ1
 */
void initPheromoneMatrix(int taskNum,int nodeNum) {
    for(int i=0; i<taskNum; i++) {
        for(int j=0; j<nodeNum; j++) {
            pheromoneMatrix(i,j)=1;
        }
    }
}

/**
 * ��������
 * @param iteratorNum ��������
 * @param antNum ��������
 */
void acaSearch(int iteratorNum,int antNum) {
    for(int itCount=0;itCount<iteratorNum;itCount++){
        //���ε����У��������ϵ�·��,�Ҹ���Ū���˾��������
        vector<MatrixXd> pathMatrix_allAnt;

        for(int antCount=0;antCount<antNum;antCount++){
            // ��antCountֻ���ϵķ������(pathMatrix[i][j]��ʾ��antCountֻ���Ͻ�i��������j�ڵ㴦��)
            //�Ƚ���ȫ����ʼ��Ϊ0
            Matrix<double,Dynamic,Dynamic> pathMatrix_oneAnt;
            for(int i=0;i<taskNum;i++){
                for(int j=0;j<antNum;j++){
                    pathMatrix_oneAnt(i,j)=0;
                }
            }
            for(int taskCount=0;taskCount<taskNum;taskCount++){
                // ����taskCount������������nodeCount���ڵ㴦��
                int nodeCount=assignOneTask(antCount, taskCount, nodes, pheromoneMatrix);
                pathMatrix_oneAnt[taskCount,nodeCount]=1;
            }
             // ����ǰ���ϵ�·������pathMatrix_allAnt
             pathMatrix_allAnt.push_back(pathMatrix_oneAnt);
        }
        // ���� ���ε����� �������� ��������ʱ��
        std::vector<int> timeArray_oneIt=calTime_oneIt(pathMatrix_allAnt);
        // �����ص����� �������ϵ� ������ʱ������ܽ����
        resultData.push_back(timeArray_oneIt);

        //������Ϣ��
        updatePheromoneMatrix(pathMatrix_allAnt, pheromoneMatrix, timeArray_oneIt);
    }
}

/**
 * ����taskCount����������ĳһ���ڵ㴦��
 * @param antCount ���ϱ��
 * @param taskCount ������
 * @param nodes �ڵ㼯��
 * @param pheromoneMatrix ��Ϣ�ؼ���
 */
 int assignOneTask(int antCount,int taskCount,int* nodes,MatrixXd pheromoneMatrix){
    // ����ǰ���ϱ�����ٽ��֮ǰ������������Ϣ�صķ��䷽ʽ
    if(antCount<=criticalPointMatrix[taskCount]){
        return maxPheromoneMatix[taskCount];
    }
    //����ǰ���ϱ�����ٽ��֮��������������ķ�ʽ
    std::mt19937 rng;
    std::uniform_int_distribution dis(0,nodeNum-1);
    return dis(rng);
 }

/**
 * ����һ�ε����У��������ϵ�������ʱ��
 * @param pathMatrix_allAnt �������ϵ�·��
 */
std::vector<int> calTime_oneIt(std::vector<MatrixXd> pathMatrix_allAnt){
    std::vector<int> time_allAnt;
    for(int antIndex=0;antIndex<pathMatrix_allAnt.size;antIndex++){
        // ��ȡ��antIndexֻ���ϵ�����·��
        MatrixXd pathMatrix=pathMatrix_allAnt[antIndex];

        //��ȡ����ʱ����Ľڵ� ��Ӧ�Ĵ���ʱ��
        int maxTime=-1;
        for(int nodeIndex=0;nodeIndex<nodeNum;nodeIndex++){
            //����ڵ�taskIndex��������ʱ��
            int time=0;
            for(int taskIndex=0;taskIndex<taskNum;taskIndex++){
                if(pathMatrix(nodeIndex)(taskIndex)==1){
                    time+=timeMatrix(nodeIndex)(taskIndex);
                }
            }
            //����maxTime
            if(time>maxTime)
                maxTime=time;
        }
        time_allAnt.push_back(maxTime);
    }
    return time_allAnt;
}

/**
 * ������Ϣ��
 * @param pathMatrix_allAnt ���ε������������ϵ�����·��
 * @param pheromoneMatrix ��Ϣ�ؾ���
 * @param timeArray_oneIt ���ε�����������ʱ��Ľ����
*/
void updatePheromoneMatrix(std::vector<MatrixXd> pathMatrix_allAnt,int *pheromoneMatrix,std::vector<int> timeArray_oneIt){
     // ������Ϣ�ؾ�˥��p%
     for(int i=0;i<taskNum;i++){
        for(int j=0;j<nodeNum;j++){
            pheromoneMatrix(i)(j)*=p;
        }
     }

     //�ҳ�������ʱ����̵����ϱ��
     int minTime=MAX_VALUE;
     int minIndex=-1;
     for(int antIndex=0;antIndex<antNum;antIndex++){
        if(timeArray_oneIt[antIndex]<minTime){
            minTime=timeArray_oneIt[antIndex];
            minIndex=antIndex;
        }
     }

    // �����ε���������·������Ϣ������q%,��ν����·�����Ǵ���ʱ����̵����ϣ��������ÿ��·��
    for(int taskIndex=0;taskIndex<taskNum;taskIndex++){
        for(int nodeIndex=0;nodeIndex<nodeNum;nodeIndex++){
            if(pathMatrix_allAnt[minIndex](taskIndex)(nodeIndex)==1){
                pheromoneMatrix(taskIndex)(nodeIndex)*=q;
            }
        }
    }
    for(int taskIndex=0;taskIndex<taskNum;taskIndex++){
        int maxPheromone=pheromoneMatrix(taskIndex)(0);
        int maxIndex=0;
        int sumPheromone=pheromoneMatrix(taskIndex)(0);
        bool isAllSame=true;

        //��ȡ�����Ϣ�ص�ֵ�������ÿ�����������Ϣ�صĺ�
        for(int nodeIndex=1;nodeIndex<nodeNum;nodeIndex++){
            if(pheromoneMatrix(taskIndex)(nodeIndex)>maxPheromone){
                maxPheromone=pheromoneMatrix(taskIndex)(nodeIndex);
                maxIndex=nodeIndex;
            }

            //����ֵ�����ھ��������һ�е�ֵ
            if(pheromoneMatrix(taskIndex)(nodeIndex)!=pheromoneMatrix(taskIndex)(nodeIndex-1)){
                isAllSame=false;
            }

            sumPheromone+=pheromoneMatrix(taskIndex)(nodeIndex);
        }
        // ��������Ϣ��ȫ����ȣ������ѡ��һ����Ϊ�����Ϣ��
        maxPheromoneMatrix.push_back(maxIndex)

        // �����ε����������ٽ��ż���criticalPointMatrix(���ٽ��֮ǰ�����ϵ����������������Ϣ��ԭ�򣬶����ٽ��֮������ϲ�������������)
        criticalPointMatrix.push_back(round(antNum*(maxPheromone/sumPheromone)));
    }
}
#endif // ANTCOLONYOPTIMIZATION_H_INCLUDED

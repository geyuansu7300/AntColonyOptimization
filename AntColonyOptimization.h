#ifndef ANTCOLONYOPTIMIZATION_H_INCLUDED
#define ANTCOLONYOPTIMIZATION_H_INCLUDED

#include <iostream>
#include<vector>
#include<Eigen/Dense>
#include<math.h>
#define MAX_VALUE 9999

using namespace std;

////////////////////////变量定义////////////////////////////
//任务数组，数组的下标表示任务的编号，数组的值表示任务的长度。比如：tasks[0]=10表示第一个任务的任务长度是10.
int *tasks;
//任务的数量，
const int taskNum=100;

//处理节点的数组。数组的下标表示处理节点的编号，数组值表示节点的处理速度
int *nodes;
//处理节点的数量
const int nodeNum=10;

//蚁群迭代的次数
int iteratorNum;
//每次迭代中蚂蚁的数量。每只蚂蚁都是一个任务调度者，每次迭代中的每一只蚂蚁都需要完成所有任务的分配，这也就是一个可行解。
int antNum;

//任务处理时间矩阵，它是一个二维矩阵。比如：timeMatrix[i][j]就表示第i个任务分配给第j个节点所需的处理时间。
//timeMatrix[i][j]=task[i]/nodes[j]
MatrixXd timeMatrix;
//pheromoneMatrix：信息素矩阵 pheromoneMatrix[i][j]=0.5就表示任务i分配给节点j这条路径上的信息素浓度为0.5
MatrixXd pheromoneMatrix;
//maxPheromoneMatrix：pheromoneMatrix矩阵的每一行中最大信息素的下标
std::vector<int> maxPheromoneMatrix[taskNum];
//criticalPointMatrix 在一次迭代中，采用随机分配策略的蚂蚁的临界编号。
std::vector<int> criticalPointMatrix[taskNum];

//p：每完成一次迭代后，信息素衰减的比例。
int p=0.5;
//q：蚂蚁每次经过一条路径，信息素增加的比例。
int q=2;
//任务处理时间的结果总集
std::vector<std::vector<int>> resultData;

void initialTasksAndNodes() {
    mt19937 rng;
    //初始化任务集合
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
////////////////////////变量定义////////////////////////////
void initTimeMatrix(int tasks[],int nodes[]);
void initPheromoneMatrix(int taskNum,int nodeNum);
void acaSearch(int iteratorNum,int antNum);
int assignOneTask(int antCount,int taskCount,int* nodes,MatrixXd pheromoneMatrix);
std::vector<int> calTime_oneIt(std::vector<MatrixXd> pathMatrix_allAnt);
void updatePheromoneMatrix(std::vector<MatrixXd> pathMatrix_allAnt,MatrixXd pheromoneMatrix,std::vector<int> timeArray_oneIt);
/***
蚁群算法
*/
void aca() {
    //初始化任务执行时间矩阵
    initTimeMatrix(tasks,nodes);
    //初始化信息素矩阵
    initPheromoneMatrix(taskNum,nodeNum);
    //迭代搜索
    acaSearch(iteratorNum,antNum);
}

/**
 * 初始化任务处理时间矩阵
 * @param tasks 任务(长度)列表
 * @param nodes 节点(处理速度)列表
 */
void initTimeMatrix(int tasks[],int nodes[]) {
    for(int i=0; i<taskNum; i++) {
        for(int j=0; j<nodeNum; j++) {
            timeMatrix(i,j)=tasks[i]/nodes[j];
        }
    }
}

/**
 * 初始化信息素矩阵(全为1)
 * @param taskNum 任务数量
 * @param nodeNum 节点数量
    全部初始化为1
 */
void initPheromoneMatrix(int taskNum,int nodeNum) {
    for(int i=0; i<taskNum; i++) {
        for(int j=0; j<nodeNum; j++) {
            pheromoneMatrix(i,j)=1;
        }
    }
}

/**
 * 迭代搜索
 * @param iteratorNum 迭代次数
 * @param antNum 蚂蚁数量
 */
void acaSearch(int iteratorNum,int antNum) {
    for(int itCount=0;itCount<iteratorNum;itCount++){
        //本次迭代中，所有蚂蚁的路径,我给它弄成了矩阵的数组
        vector<MatrixXd> pathMatrix_allAnt;

        for(int antCount=0;antCount<antNum;antCount++){
            // 第antCount只蚂蚁的分配策略(pathMatrix[i][j]表示第antCount只蚂蚁将i任务分配给j节点处理)
            //先将其全部初始化为0
            Matrix<double,Dynamic,Dynamic> pathMatrix_oneAnt;
            for(int i=0;i<taskNum;i++){
                for(int j=0;j<antNum;j++){
                    pathMatrix_oneAnt(i,j)=0;
                }
            }
            for(int taskCount=0;taskCount<taskNum;taskCount++){
                // 将第taskCount个任务分配给第nodeCount个节点处理
                int nodeCount=assignOneTask(antCount, taskCount, nodes, pheromoneMatrix);
                pathMatrix_oneAnt[taskCount,nodeCount]=1;
            }
             // 将当前蚂蚁的路径加入pathMatrix_allAnt
             pathMatrix_allAnt.push_back(pathMatrix_oneAnt);
        }
        // 计算 本次迭代中 所有蚂蚁 的任务处理时间
        std::vector<int> timeArray_oneIt=calTime_oneIt(pathMatrix_allAnt);
        // 将本地迭代中 所有蚂蚁的 任务处理时间加入总结果集
        resultData.push_back(timeArray_oneIt);

        //更新信息素
        updatePheromoneMatrix(pathMatrix_allAnt, pheromoneMatrix, timeArray_oneIt);
    }
}

/**
 * 将第taskCount个任务分配给某一个节点处理
 * @param antCount 蚂蚁编号
 * @param taskCount 任务编号
 * @param nodes 节点集合
 * @param pheromoneMatrix 信息素集合
 */
 int assignOneTask(int antCount,int taskCount,int* nodes,MatrixXd pheromoneMatrix){
    // 若当前蚂蚁编号在临界点之前，则采用最大信息素的分配方式
    if(antCount<=criticalPointMatrix[taskCount]){
        return maxPheromoneMatix[taskCount];
    }
    //若当前蚂蚁编号在临界点之后，则采用随机分配的方式
    std::mt19937 rng;
    std::uniform_int_distribution dis(0,nodeNum-1);
    return dis(rng);
 }

/**
 * 计算一次迭代中，所有蚂蚁的任务处理时间
 * @param pathMatrix_allAnt 所有蚂蚁的路径
 */
std::vector<int> calTime_oneIt(std::vector<MatrixXd> pathMatrix_allAnt){
    std::vector<int> time_allAnt;
    for(int antIndex=0;antIndex<pathMatrix_allAnt.size;antIndex++){
        // 获取第antIndex只蚂蚁的行走路径
        MatrixXd pathMatrix=pathMatrix_allAnt[antIndex];

        //获取处理时间最长的节点 对应的处理时间
        int maxTime=-1;
        for(int nodeIndex=0;nodeIndex<nodeNum;nodeIndex++){
            //计算节点taskIndex的任务处理时间
            int time=0;
            for(int taskIndex=0;taskIndex<taskNum;taskIndex++){
                if(pathMatrix(nodeIndex)(taskIndex)==1){
                    time+=timeMatrix(nodeIndex)(taskIndex);
                }
            }
            //更新maxTime
            if(time>maxTime)
                maxTime=time;
        }
        time_allAnt.push_back(maxTime);
    }
    return time_allAnt;
}

/**
 * 更新信息素
 * @param pathMatrix_allAnt 本次迭代中所有蚂蚁的行走路径
 * @param pheromoneMatrix 信息素矩阵
 * @param timeArray_oneIt 本次迭代的任务处理时间的结果集
*/
void updatePheromoneMatrix(std::vector<MatrixXd> pathMatrix_allAnt,int *pheromoneMatrix,std::vector<int> timeArray_oneIt){
     // 所有信息素均衰减p%
     for(int i=0;i<taskNum;i++){
        for(int j=0;j<nodeNum;j++){
            pheromoneMatrix(i)(j)*=p;
        }
     }

     //找出任务处理时间最短的蚂蚁编号
     int minTime=MAX_VALUE;
     int minIndex=-1;
     for(int antIndex=0;antIndex<antNum;antIndex++){
        if(timeArray_oneIt[antIndex]<minTime){
            minTime=timeArray_oneIt[antIndex];
            minIndex=antIndex;
        }
     }

    // 将本次迭代中最优路径的信息素增加q%,所谓最优路径就是处理时间最短的蚂蚁，它分配的每个路径
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

        //获取最大信息素的值、坐标和每个任务最大信息素的和
        for(int nodeIndex=1;nodeIndex<nodeNum;nodeIndex++){
            if(pheromoneMatrix(taskIndex)(nodeIndex)>maxPheromone){
                maxPheromone=pheromoneMatrix(taskIndex)(nodeIndex);
                maxIndex=nodeIndex;
            }

            //若该值不等于矩阵该行上一列的值
            if(pheromoneMatrix(taskIndex)(nodeIndex)!=pheromoneMatrix(taskIndex)(nodeIndex-1)){
                isAllSame=false;
            }

            sumPheromone+=pheromoneMatrix(taskIndex)(nodeIndex);
        }
        // 若本行信息素全都相等，则随机选择一个作为最大信息素
        maxPheromoneMatrix.push_back(maxIndex)

        // 将本次迭代的蚂蚁临界编号加入criticalPointMatrix(该临界点之前的蚂蚁的任务分配根据最大信息素原则，而该临界点之后的蚂蚁采用随机分配策略)
        criticalPointMatrix.push_back(round(antNum*(maxPheromone/sumPheromone)));
    }
}
#endif // ANTCOLONYOPTIMIZATION_H_INCLUDED

#include<cstdio>
#include<iostream>
#include<vector>
using namespace std;
void Ctree() {
	hello();
}
void Ctree(BTNode* b, SqBTree a, int i)
{
	if (b != NULL)
	{
		a[i] = b->data;
		Ctree(b->lchild, a, 2 * i);
		Ctree(b->rchild, a, 2 * i + 1);
	}
	else a[i] = '#';
}

void FindMinNode(BTNode* b, char& min)
{
	if (b->data < min)
		min = b->data;
	FindMinNode(b->lchild, min); //在左子树中找最小结点值
	FindMinNode(b->rchild, min); //在右子树中找最小结点值
}
void MinNode(BTNode* b) //输出最小结点值
{
	if (b != NULL)
	{
		char min = b->data;
		FindMinNode(b, min);
		cout<<"Min="<<min<<endl;
	}
}
void PrintNode(BTNode* p) //输出以p为根结点的子树
{
	if (p != NULL)
	{
		cout<<p->data;
		PrintNode(p->lchild);
		PrintNode(p->rchild);
	}
}
void X_Child(BTNode* b, char x) //输出x结点的子孙
{
	if (b != NULL)
	{
		if (b->data == x)
		{
			if (b->lchild != NULL)
				PrintNode(b->lchild);
			if (b->rchild != NULL)
				PrintNode(b->rchild);
			return;
		}
		X_Child(b->lchild, x);
		X_Child(b->rchild, x);
	}
}
bool CompBTree(BTNode* b)
{
	BTNode* Qu[MaxSize], * p; //定义一个队列，用于层次遍历
	int front = 0, rear = 0; //环形队列的队头队尾指针
	bool cm = true; //cm为真表示二叉树为完全二叉树
	bool bj = true; //bj为真表示到目前为止所有结点均有左右孩子
	if (b == NULL) return true; //空树当成特殊的完全二叉树
	rear++;
	Qu[rear] = b; //根结点进队
	while (front != rear) //队列不空
	{
		front = (front + 1) % MaxSize;
		p = Qu[front]; //出队结点p
		if (p->lchild == NULL) //p结点没有左孩子
		{
			bj = false; //出现结点p缺左孩子的情况
			if (p->rchild != NULL) //没有左孩子但有右孩子,违反(1),
				cm = false;
		}
		else //p结点有左孩子
		{
			if (!bj) cm = false; //bj为假而结点p还有左孩子，违反(2)
			rear = (rear + 1) % MaxSize;
			Qu[rear] = p->lchild; //左孩子进队
			if (p->rchild == NULL)
				bj = false; //出现结点p缺右孩子的情况
			else //p有左右孩子,则继续判断
			{
				rear = (rear + 1) % MaxSize;
				Qu[rear] = p->rchild; //将p结点的右孩子进队
			}
		}
	}
	return cm;
}
void InDs1(MatGraph g) //求出图G 中每个顶点的入度
{
	int i, j, n;
	printf("各顶点入度:\n");
	for (j = 0;j < g.n;j++)//入度为矩阵中第j行元素之和
	{
		n = 0;
		for (i = 0;i < g.n;i++)
			if (g.edges[i][j] != 0)
				n++; //n累计入度数
		printf(" 顶点%d:%d\n", j, n);
	}
}
void OutDs1(MatGraph g) //求出图G中每个顶点的出度
{
	int i, j, n;
	printf("各顶点出度:\n");
	for (i = 0;i < g.n;i++)//出度为矩阵中第i行元素之和
	{
		n = 0;
		for (j = 0;j < g.n;j++)
			if (g.edges[i][j] != 0)
				n++; //n累计出度数
		printf(" 顶点%d:%d\n", i, n);
	}
}
void ZeroOutDs1(MatGraph g) //求出图G中出度为0的顶点个数
{
	int i, j, n;
	printf("出度为0的顶点:");
	for (i = 0;i < g.n;i++)
	{
		n = 0;
		for (j = 0;j < g.n;j++)
			if (g.edges[i][j] != 0) //存在一条出边
				n++;
		if (n == 0)
			printf("%2d\n", i);
	}
	printf("\n");
}



void InDs2(AdjGraph* G) //求出图G中每个顶点的入度
{
	ArcNode* p;
	int A[MAXV], i; //A存放各顶点的入度
	for (i = 0;i < G->n;i++) //A中元素置初值0
		A[i] = 0;
	for (i = 0;i < G->n;i++) //扫描所有头结点
	{
		p = G->adjlist[i].firstarc;
		while (p != NULL) //扫描边结点
		{
			A[p->adjvex]++; //表示i到p->adjvex顶点有一条边
			p = p->nextarc;
		}
	}
	printf("各顶点入度:\n"); //输出各顶点的入度
	for (i = 0;i < G->n;i++)
		printf(" 顶点%d:%d\n", i, A[i]);
}
void OutDs2(AdjGraph* G) //求出图G中每个顶点的出度
{
	int i, n;
	ArcNode* p;
	printf("各顶点出度:\n");
	for (i = 0;i < G->n;i++) //扫描所有头结点
	{
		n = 0;
		p = G->adjlist[i].firstarc;
		while (p != NULL) //扫描边结点
		{
			n++; //累计出边的数
			p = p->nextarc;
		}
		printf(" 顶点%d:%d\n", i, n);
	}
}
void ZeroOutDs2(AdjGraph* G) //求出图G中出度为0的顶点数
{
	int i, n;
	ArcNode* p;
	printf("出度为0的顶点:");
	for (i = 0;i < G->n;i++) //扫描所有头结点
	{
		p = G->adjlist[i].firstarc;
		n = 0;
		while (p != NULL) //扫描边结点
		{
			n++; //累计出边的数
			p = p->nextarc;
		}
		if (n == 0) //输出出边数为0的顶点编号
			printf("%2d", i);
	}
	printf("\n");
}

int visited[MAXV];
void findpath(AdjGraph* G, int u, int v, int path[], int d, int length)
{ //d表示path中顶点个数，初始为0；length表示路径长度，初始为0
	int w, i;
	ArcNode* p;
	path[d] = u; d++; //顶点u加入到路径中，d增1
	visited[u] = 1; //置已访问标记
	if (u == v && d > 0) //找到一条路径则输出
	{
		printf(" 路径长度:%d, 路径:", length);
		for (i = 0;i < d;i++)
			printf("%2d", path[i]);
		printf("\n");
	}
	p = G->adjlist[u].firstarc; //p指向顶点u的第一个邻接点
	while (p != NULL)
	{
		w = p->adjvex; //w为顶点u的邻接点
		if (visited[w] == 0) //若w顶点未访问,递归访问它
			findpath(G, w, v, path, d, p->weight + length);
		p = p->nextarc; //p指向顶点u的下一个邻接点
	}
	visited[u] = 0; //恢复环境,使该顶点可重新使用
}
int main() {
	AdjGraph* G;
	int A[MAXV][MAXV] = {
	{0,4,6,6,INF,INF,INF}, 
	{INF,0,1,INF,7,INF,INF},
	{INF,INF,0,INF,6,4,INF},
	{INF,INF,2,0,INF,5,INF},
	{INF,INF,INF,INF,0,INF,6},
	{INF,INF,INF,INF,1,0,8},
	{INF,INF,INF,INF,INF,INF,0} };
		
	int n = 7, e = 12;
	CreateAdj(G, A, n, e); //创建《教程》中图8.35的邻接表
	printf("图G的邻接表:\n");
	DispAdj(G); //输出邻接表
	int u = 0, v = 5;
	int path[MAXV];
	printf("从%d->%d的所有路径:\n", u, v);
	findpath(G, u, v, path, 0, 0);
	DestroyAdj(G);
	return 1;
}
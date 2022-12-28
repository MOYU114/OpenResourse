#include"seqlist.h"
void swap(RecType& x, RecType& y)	//x和y交换
{
	RecType tmp = x;
	x = y; y = tmp;
}

void CreateList(RecType R[], KeyType keys[], int n)	//创建顺序表
{
	for (int i = 0;i < n;i++)			//R[0..n-1]存放排序记录
		R[i].key = keys[i];
}
void DispList(RecType R[], int n)	//输出顺序表
{
	for (int i = 0;i < n;i++)
		printf("%d ", R[i].key);
	printf("\n");
}
//----以下运算针对堆排序的程序
void CreateList1(RecType R[], KeyType keys[], int n)	//创建顺序表
{
	for (int i = 1;i <= n;i++)			//R[1..n]存放排序记录
		R[i].key = keys[i - 1];
}
void DispList1(RecType R[], int n)	//输出顺序表
{
	for (int i = 1;i <= n;i++)
		printf("%d ", R[i].key);
	printf("\n");
}
#include<cstdio>
#include<iostream>
#include<stdlib.h>
#define N 10 //数组大小（可更改）
using namespace std;
int max(int a, int b, int c)
{
	return a > b ? a > c ? a : c : b > c ? b : c;
}
//循环算法
int loopprocess(int a[])
{
	int ThisSum, result = 0;
	int i, j, k;
	for (i = 0;i < N;i++) {
		for (j = i;j < N;j++) {
			ThisSum = 0;
			for (k = i;k <= j;k++) {   //从i到j求其和
				ThisSum += a[k];
			}
			if (ThisSum > result) {    //判断现在的结果和现在的和的大小，若现在和大则更新
				result = ThisSum;
			}
		}
	}
	return result;
}
//分而治之，即从中间分开，分别比较左侧最大子列和，右侧最大子列和，跨中线最大子列和，选出最大的作为结果
int devideprocess(int a[], int left, int right)
{
	int half, i;
	int MaxLeftSum, MaxRightSum;         //分别存储左边、右边的最大子列和
	int ThisMaxLeftSum, ThisMaxRightSum;
	int ThisLeftSum, ThisRightSum;
	if (left == right)                 //若左等于右，证明其子列只剩一个元素，已经到了出口位置
	{
		if (a[left] > 0) {           //将大于零的数返回，否则返回0来将其舍去
			return a[left];
		}
		else {
			return 0;
		}
	}
	half = (left + right) / 2;       //取中间的数而分开处理
	MaxLeftSum = devideprocess(a, left, half);
	MaxRightSum = devideprocess(a, half + 1, right);  //使用递归处理分开的两部分

	//求跨越中线的最大子列和
	ThisMaxLeftSum = 0;
	ThisLeftSum = 0;
	for (i = half;i >= left;i--) {  //从中线左边开始计算
		ThisLeftSum += a[i];
		if (ThisLeftSum > ThisMaxLeftSum) {
			ThisMaxLeftSum = ThisLeftSum;
		}
	}
	ThisMaxRightSum = 0;
	ThisRightSum = 0;
	for (i = half + 1;i <= right;i++) { //从中线右边开始计算
		ThisRightSum += a[i];
		if (ThisRightSum > ThisMaxRightSum) {
			ThisMaxRightSum = ThisRightSum;
		}
	}
	return max(MaxLeftSum, MaxRightSum, ThisMaxRightSum + ThisMaxLeftSum);
}
//在线处理
int lineprocess(int a[])
{
	int ThisSum = 0, i = 0, result = 0;
	for (i = 0;i < N;i++)
	{
		ThisSum += a[i];//计算总和
		if (ThisSum >= result)//若得出的总和比原来记录的总和大，使结果更新
		{
			result = ThisSum;
		}
		else if (ThisSum < 0)//若现在的总和已小于0，则再加数只会让之更小
		{
			ThisSum = 0;
		}
	}
	return result;
}
int main()
{
	int i, resultdev, resultline, resultloop;
	int a[N];
	for (i = 0;i < N;i++)
	{
		a[i] = rand() % 2000 - 1000;
	}
	resultloop = loopprocess(a);
	cout << resultloop <<" ";
	resultdev = devideprocess(a,0,N-1);
	cout << resultdev << " ";
	resultline = lineprocess(a);
	cout << resultline << " ";

}

#include<cstdio>
#include<iostream>
#define N 10
using namespace std;
int max(int a, int b, int c)
{
	return a > b ? a > c ? a : c : b > c ? b : c;
}
//ѭ���㷨
int loopprocess(int a[]) 
{
	int ThisSum, result = 0;
	int i, j, k;
	for (i = 0 ;i < N;i++) {
		for (j = i;j < N;j++) {
			ThisSum = 0;
			for (k = i;k <= j;k++) {   //��i��j�����
				ThisSum += a[k];
			}
			if (ThisSum > result) {    //�ж����ڵĽ�������ڵĺ͵Ĵ�С�������ںʹ������
				result = ThisSum;
			}
		}
	}
	return result;
}
//�ֶ���֮
int devideprocess(int a[],int left,int right)
{
	int half,i;
	int MaxLeftSum, MaxRightSum;
	int ThisMaxLeftSum, ThisMaxRightSum;
	int ThisLeftSum, ThisRightSum;
	if (left = right) \
	{
		if (a[left] > 0) {
			return a[left];
		}
		else {
			return 0;
		}
	}
	half = (left + right) / 2;
	MaxLeftSum = devideprocess(a, left, half);
	MaxRightSum = devideprocess(a, half + 1, right);

	ThisMaxLeftSum = 0;
	ThisLeftSum = 0;
	for (i = half;i >= left;i--) {
		ThisLeftSum += a[i];
		if(ThisLeftSum>ThisMaxLeftSum){
			ThisMaxLeftSum = ThisLeftSum;
		}
	}
	ThisMaxRightSum = 0;
	ThisRightSum = 0;
	for (i = half+1;i <= right;i++) {
		ThisRightSum += a[i];
		if (ThisRightSum > ThisMaxRightSum) {
			ThisMaxRightSum = ThisRightSum;
		}
	}
	return max(MaxLeftSum, MaxRightSum, ThisMaxRightSum+ ThisMaxLeftSum);
}
//���ߴ���
int lineprocess(int a[]) 
{
	int ThisSum=0,i=0,result=0;
	for (i=0;i < N;i++) 
	{
		ThisSum += a[i];//�����ܺ�
		if(ThisSum>=result)//���ó����ܺͱ�ԭ����¼���ܺʹ�ʹ�������
		{
			result = ThisSum;
		}
		else if(ThisSum<0)//�����ڵ��ܺ���С��0�����ټ���ֻ����֮��С
		{
			ThisSum = 0;
		}
	}
	return result;
}
int main()
{
	int i,resultdev,resultline,resultloop;
	int a[N];
	int b[10] = { -1,3,-2,4,-6,1,6,-1,0,0 };
	for (i = 0;i < N;i++)
	{
		a[i] = rand() % 2000 - 1000;
	}
	resultloop = loopprocess(b);
	cout << resultloop;
	//resultdev = devideprocess(b,0,N-1);
	//cout << resultdev;	
	resultline=lineprocess(b);
	cout << resultline;
	
}
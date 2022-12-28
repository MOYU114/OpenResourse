#include<cstdio>
#include<iostream>
#include<stdlib.h>
#define N 10 //�����С���ɸ��ģ�
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
	for (i = 0;i < N;i++) {
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
//�ֶ���֮�������м�ֿ����ֱ�Ƚ����������кͣ��Ҳ�������кͣ�������������кͣ�ѡ��������Ϊ���
int devideprocess(int a[], int left, int right)
{
	int half, i;
	int MaxLeftSum, MaxRightSum;         //�ֱ�洢��ߡ��ұߵ�������к�
	int ThisMaxLeftSum, ThisMaxRightSum;
	int ThisLeftSum, ThisRightSum;
	if (left == right)                 //��������ң�֤��������ֻʣһ��Ԫ�أ��Ѿ����˳���λ��
	{
		if (a[left] > 0) {           //��������������أ����򷵻�0��������ȥ
			return a[left];
		}
		else {
			return 0;
		}
	}
	half = (left + right) / 2;       //ȡ�м�������ֿ�����
	MaxLeftSum = devideprocess(a, left, half);
	MaxRightSum = devideprocess(a, half + 1, right);  //ʹ�õݹ鴦��ֿ���������

	//���Խ���ߵ�������к�
	ThisMaxLeftSum = 0;
	ThisLeftSum = 0;
	for (i = half;i >= left;i--) {  //��������߿�ʼ����
		ThisLeftSum += a[i];
		if (ThisLeftSum > ThisMaxLeftSum) {
			ThisMaxLeftSum = ThisLeftSum;
		}
	}
	ThisMaxRightSum = 0;
	ThisRightSum = 0;
	for (i = half + 1;i <= right;i++) { //�������ұ߿�ʼ����
		ThisRightSum += a[i];
		if (ThisRightSum > ThisMaxRightSum) {
			ThisMaxRightSum = ThisRightSum;
		}
	}
	return max(MaxLeftSum, MaxRightSum, ThisMaxRightSum + ThisMaxLeftSum);
}
//���ߴ���
int lineprocess(int a[])
{
	int ThisSum = 0, i = 0, result = 0;
	for (i = 0;i < N;i++)
	{
		ThisSum += a[i];//�����ܺ�
		if (ThisSum >= result)//���ó����ܺͱ�ԭ����¼���ܺʹ�ʹ�������
		{
			result = ThisSum;
		}
		else if (ThisSum < 0)//�����ڵ��ܺ���С��0�����ټ���ֻ����֮��С
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

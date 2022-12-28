#include<iostream>
using namespace std;
#define MAXL 100
#define KeyType int
#define InfoType int
typedef struct {
	KeyType key;		//KeyTypeΪ�ؼ��ֵ���������
	InfoType data;		//����������
}RecType;			//����˳���Ԫ������
int SqSearch(RecType R[],int n,KeyType k) {
	int i = 0;
	for (i = 0;i < n;i++)
		if (R[i].key == k) return i+1;
	
	return 0;
}
int BinSearch(RecType R[], int n, KeyType k) {
	int low = R[0].key;
	int high = R[n - 1].key;
	while (low <= high) {
	int mid = (low + high) / 2;
	if (R[mid].key == k)return mid + 1;
	if (R[mid].key > k)return low = mid + 1;
	else high = mid - 1;
	}
	return 0;
}
typedef struct node
{
	KeyType key;            	  //�ؼ�����
	InfoType data;          	  //����������
	struct node* lchild, rchild; 	  //���Һ���ָ��
}  BSTNode;

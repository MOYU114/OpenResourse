#include<iostream>
using namespace std;
#define MAXL 100
#define KeyType int
#define InfoType int
typedef struct {
	KeyType key;		//KeyType为关键字的数据类型
	InfoType data;		//其他数据项
}RecType;			//查找顺序表元素类型
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
	KeyType key;            	  //关键字项
	InfoType data;          	  //其他数据域
	struct node* lchild, rchild; 	  //左右孩子指针
}  BSTNode;

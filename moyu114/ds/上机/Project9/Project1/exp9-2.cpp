#include"sqlist.h"
typedef int KeyType;
int BinSearch(SqList* S, KeyType k) {
	int low=0, high=S->length-1, mid,cnt=0;
	while (low <= high) {
		mid = (low + high) / 2;
		cnt++;
		printf("��%d�Ƚϣ���S[%d,%d]�в���Ԫ��S[%d]",cnt,low,high,mid);
		if (S->data[mid] == k) {
			cout << "�Ѳ�ѯ��!" << endl;
			return mid + 1;
		}else if (S->data[mid] >k) {
			cout << "�Ȳ���Ԫ�ش�" << endl;
			high = mid - 1;
		}
		else {
			cout << "�Ȳ���Ԫ��С" << endl;
			low = mid + 1;
		}
	}
}
int main() {
	SqList* s;
	KeyType k = 9;
	int result = -1;
	ElemType a[] = { 1,2,3,4,5,6,7,8,9 };
	InitList(s);
	CreateList(s,a,9);
	 result = BinSearch(s,k);
	 if (result != -1) {
		 printf("����Ԫ��%d��˳����%d��\n", k, result);
	}
}
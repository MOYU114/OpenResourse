//˳�����������㷨
#include <iostream>
using namespace std;
#define MAXL 100		//��󳤶�
typedef int KeyType;	//����ؼ�������Ϊint
typedef char InfoType;

typedef struct
{
	KeyType key;		//�ؼ�����
	InfoType data;		//�������������ΪInfoType
} RecType;				//����Ԫ�ص�����

void swap(RecType& x, RecType& y);	
void CreateList(RecType R[], KeyType keys[], int n);
void DispList(RecType R[], int n);
void CreateList1(RecType R[], KeyType keys[], int n);
void DispList1(RecType R[], int n);
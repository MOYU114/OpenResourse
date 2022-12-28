#include"btree3.h"
void CreateBTNode(BTNode*& b, char* str)
{      //��str�Ƴ�������b
    BTNode* St[MaxSize], * p;//����ջ

    int top = -1, k, j = 0;
    char ch;
    b = NULL;		//�����Ķ�������ʼʱΪ��
    ch = str[j];
    p = (BTNode*)malloc(sizeof(BTNode));
    p->data = ch;  p->lchild = p->rchild = NULL;
    while (ch != '\0')  	//strδɨ����ʱѭ��
    {
        switch (ch)
        {
        case '(': top++; St[top] = p; k = 1; break;	//���������ӽ�㣬��ջ
        case ')': top--;  break;
        case ',': k = 2;  break; 			//����Ϊ�Һ��ӽ��
        default:        		 //�������ֵ
            p = (BTNode*)malloc(sizeof(BTNode));
            p->data = ch;  p->lchild = p->rchild = NULL;
            if (b == NULL)    	//pΪ�������ĸ����
                b = p;
            else    			//�ѽ��������������
            {
                switch (k)
                {
                case 1:  St[top]->lchild = p;  break;
                case 2:  St[top]->rchild = p;  break;
                }
            }
        }
        j++;  ch = str[j];		//����ɨ��str
    }
}
void DestroyBT(BTNode*& b)
{
    if (b == NULL) return;
    else
    {
        DestroyBT(b->lchild);//����������
        DestroyBT(b->rchild);//����������
        free(b);      //ֻʣ��һ�����*b��ֱ���ͷ�
    }
}
BTNode* FindNode(BTNode* b, ElemType x)
{
    BTNode* p;
    //��������
    if (b == NULL) return NULL;//bΪ��
    else if (b->data == x) return b;//�ҵ���ֵ
    //��������
    else
    {
        p = FindNode(b->lchild, x);//Ѱ��b���������Ƿ�������ֵ
        if (p != NULL) return p;//��Ϊ�գ�֤���У�����p
        else return FindNode(b->rchild, x);//����Ѱ��������
    }
}
BTNode* LchildNode(BTNode* p)
{
    return p->lchild;
}

BTNode* RchildNode(BTNode* p)
{
    return p->rchild;
}
int BTNodeDepth(BTNode* b)
{
    int lchilddep, rchilddep;
    if (b == NULL) return(0); 	//�����ĸ߶�Ϊ0
    else
    {
        lchilddep = BTNodeDepth(b->lchild);
        //���������ĸ߶�Ϊlchilddep
        rchilddep = BTNodeDepth(b->rchild);
        //���������ĸ߶�Ϊrchilddep
        return(lchilddep > rchilddep) ? (lchilddep + 1) : (rchilddep + 1);
    }
}
void DispBTNode(BTNode* b)
{
    if (b != NULL)
    {
        printf("%c", b->data);
        if (b->lchild != NULL || b->rchild != NULL)
        {
            printf("(");
            DispBTNode(b->lchild);//�ݹ鴦��������
            if (b->rchild != NULL) printf("��");
            DispBTNode(b->rchild);//�ݹ鴦��������
            printf(")");
        }
    }
}

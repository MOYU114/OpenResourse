#include"btree.h"

int main() {
    char str[] = { "A(B(D,E(H(J,K(L,M(,N))))),C(F,G(,I)))" };
    BTNode* b, * p, * lp, * rp;
    cout << "��1������������" << endl;
    CreateBTNode(b, str);
    cout << "��2�����������b" << endl;
    DispBTNode(b);
    cout << endl;
    cout << "��3�����'H'�������Һ���ֵ" << endl;
    p = FindNode(b, 'H');
    if (p != NULL) {
        lp = LchildNode(p);
        rp = RchildNode(p);
        if (lp != NULL)
            cout << "H������Ϊ��" << lp->data<<endl;
        else
            cout<< "H������" << endl;
        if (rp != NULL)
            cout << "H���Һ���Ϊ��" << rp->data << endl;
        else
            cout << "H���Һ���" << endl;
    }
    else
        cout << "H�����Һ���" << endl;
    cout << "��4�����������b�ĸ߶�" << endl;
    cout << "b�ĸ߶�Ϊ��" << BTNodeDepth(b) << endl;
    cout << "��5���ͷŶ�����b" << endl;
    DestroyBT(b);
}
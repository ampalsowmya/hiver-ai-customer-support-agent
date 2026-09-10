314\site-packages\pandas\core\generic.py", line 3988, in to_csv  
    return DataFrameRenderer(formatter).to_csv(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        path_or_buf,
        ^^^^^^^^^^^^
    ...<14 lines>...
        storage_options=storage_options,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "C:\Users\SOWMYA KARTHIKEYAN\AppData\Roaming\Python\Python314\site-packages\pandas\io\formats\format.py", line 1025, in to_csv
    csv_formatter.save()
    ~~~~~~~~~~~~~~~~~~^^
  File "C:\Users\SOWMYA KARTHIKEYAN\AppData\Roaming\Python\Python314\site-packages\pandas\io\formats\csvs.py", line 251, in save  
    with get_handle(
         ~~~~~~~~~~^
        self.filepath_or_buffer,
        ^^^^^^^^^^^^^^^^^^^^^^^^
    ...<4 lines>...
        storage_options=self.storage_options,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ) as handles:
    ^
  File "C:\Users\SOWMYA KARTHIKEYAN\AppData\Roaming\Python\Python314\site-packages\pandas\io\common.py", line 926, in get_handle  
    handle = open(
        handle,
    ...<3 lines>...
        newline="",
    )
PermissionError: [Errno 13] Permission denied: 'outputs/evaluation/human_agreement.csv'
PS C:\Users\SOWMYA KARTHIKEYAN\OneDrive\Documents\Desktop\HIVER>Remove-Item outputs\evaluation\human_agreement.csv -Force
Remove-Item : Cannot remove item C:\Users\SOWMYA KARTHIKEYAN\One
Drive\Documents\Desktop\HIVER\outputs\evaluation\human_agreement 
.csv: The process cannot access the file 'C:\Users\SOWMYA KARTHI 
KEYAN\OneDrive\Documents\Desktop\HIVER\outputs\evaluation\human_ 
agreement.csv' because it is being used by another process.      
At line:1 char:1
+ Remove-Item outputs\evaluation\human_agreement.csv -Force      
+ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~      
    + CategoryInfo          : WriteError: (C:\Users\SOWMYA...n_  
   agreement.csv:FileInfo) [Remove-Item], IOException
    + FullyQualifiedErrorId : RemoveFileSystemItemIOError,Micro  
   soft.PowerShell.Commands.RemoveItemCommand
PS C:\Users\SOWMYA KARTHIKEYAN\OneDrive\Documents\Desktop\HIVER>Remove-Item outputs\evaluation\human_agreement.csv -Force
Remove-Item : Cannot remove item C:\Users\SOWMYA KARTHIKEYAN\One
Drive\Documents\Desktop\HIVER\outputs\evaluation\human_agreement 
.csv: The process cannot access the file 'C:\Users\SOWMYA KARTHI 
KEYAN\OneDrive\Documents\Desktop\HIVER\outputs\evaluation\human_ 
agreement.csv' because it is being used by another process.      
At line:1 char:1
+ Remove-Item outputs\evaluation\human_agreement.csv -Force      
+ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~      
    + CategoryInfo          : WriteError: (C:\Users\SOWMYA...n_  
   agreement.csv:FileInfo) [Remove-Item], IOException
    + FullyQualifiedErrorId : RemoveFileSystemItemIOError,Micro  
   soft.PowerShell.Commands.RemoveItemCommand
PS C:\Users\SOWMYA KARTHIKEYAN\OneDrive\Documents\Desktop\HIVER>python -c "import pandas as pd; a=pd.read_csv('outputs/evaluation/llm_judge_evaluation.csv'); h=[3,1,2,4,5,5,4,2,3,4,5,3,2,3,4,5,3,4,5,4,2,1,2,2,3,3,1,2,1,3]; a['human_overall']=h; a.to_csv('outputs/evaluation/human_agreement.csv',index=False); print('SUCCESS: 30 human scores saved')"
Traceback (most recent call last):
  File "<string>", line 1, in <module>
    import pandas as pd; a=pd.read_csv('outputs/evaluation/llm_judge_evaluation.csv'); h=[3,1,2,4,5,5,4,2,3,4,5,3,2,3,4,5,3,4,5,4,2,1,2,2,3,3,1,2,1,3]; a['human_overall']=h; a.to_csv('outputs/evaluation/human_agreement.csv',index=False); print('SUCCESS: 30 human scores saved')
                                                                 
                                                                 
                                            ~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\SOWMYA KARTHIKEYAN\AppData\Roaming\Python\Python314\site-packages\pandas\core\generic.py", line 3988, in to_csv  
    return DataFrameRenderer(formatter).to_csv(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        path_or_buf,
        ^^^^^^^^^^^^
    ...<14 lines>...
        storage_options=storage_options,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "C:\Users\SOWMYA KARTHIKEYAN\AppData\Roaming\Python\Python314\site-packages\pandas\io\formats\format.py", line 1025, in to_csv
    csv_formatter.save()
    ~~~~~~~~~~~~~~~~~~^^
  File "C:\Users\SOWMYA KARTHIKEYAN\AppData\Roaming\Python\Python314\site-packages\pandas\io\formats\csvs.py", line 251, in save  
    with get_handle(
         ~~~~~~~~~~^
        self.filepath_or_buffer,
        ^^^^^^^^^^^^^^^^^^^^^^^^
    ...<4 lines>...
        storage_options=self.storage_options,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ) as handles:
    ^
  File "C:\Users\SOWMYA KARTHIKEYAN\AppData\Roaming\Python\Python314\site-packages\pandas\io\common.py", line 926, in get_handle  
    handle = open(
        handle,
    ...<3 lines>...
        newline="",
    )
PermissionError: [Errno 13] Permission denied: 'outputs/evaluation/human_agreement.csv'
PS C:\Users\SOWMYA KARTHIKEYAN\OneDrive\Documents\Desktop\HIVER>taskkill /F /IM EXCEL.EXE
ERROR: The process "EXCEL.EXE" not found.
PS C:\Users\SOWMYA KARTHIKEYAN\OneDrive\Documents\Desktop\HIVER>python -c "import pandas as pd; a=pd.read_csv('outputs/evaluation/llm_judge_evaluation.csv'); h=[3,1,2,4,5,5,4,2,3,4,5,3,2,3,4,5,3,4,5,4,2,1,2,2,3,3,1,2,1,3]; a['human_overall']=h; a.to_csv('outputs/evaluation/human_agreement.csv',index=False); print('SUCCESS: 30 human scores saved')"
Traceback (most recent call last):
  File "<string>", line 1, in <module>
    import pandas as pd; a=pd.read_csv('outputs/evaluation/llm_judge_evaluation.csv'); h=[3,1,2,4,5,5,4,2,3,4,5,3,2,3,4,5,3,4,5,4,2,1,2,2,3,3,1,2,1,3]; a['human_overall']=h; a.to_csv('outputs/evaluation/human_agreement.csv',index=False); print('SUCCESS: 30 human scores saved')
                                                                 
                                                                 
                                            ~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\SOWMYA KARTHIKEYAN\AppData\Roaming\Python\Python314\site-packages\pandas\core\generic.py", line 3988, in to_csv  
    return DataFrameRenderer(formatter).to_csv(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        path_or_buf,
        ^^^^^^^^^^^^
    ...<14 lines>...
        storage_options=storage_options,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "C:\Users\SOWMYA KARTHIKEYAN\AppData\Roaming\Python\Python314\site-packages\pandas\io\formats\format.py", line 1025, in to_csv
    csv_formatter.save()
    ~~~~~~~~~~~~~~~~~~^^
  File "C:\Users\SOWMYA KARTHIKEYAN\AppData\Roaming\Python\Python314\site-packages\pandas\io\formats\csvs.py", line 251, in save  
    with get_handle(
         ~~~~~~~~~~^
        self.filepath_or_buffer,
        ^^^^^^^^^^^^^^^^^^^^^^^^
    ...<4 lines>...
        storage_options=self.storage_options,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ) as handles:
    ^
  File "C:\Users\SOWMYA KARTHIKEYAN\AppData\Roaming\Python\Python314\site-packages\pandas\io\common.py", line 926, in get_handle  
    handle = open(
        handle,
    ...<3 lines>...
        newline="",
    )
PermissionError: [Errno 13] Permission denied: 'outputs/evaluation/human_agreement.csv'
PS C:\Users\SOWMYA KARTHIKEYAN\OneDrive\Documents\Desktop\HIVER>python -c "import pandas as pd; from sklearn.metrics import cohen_kappa_score; a=pd.read_csv('outputs/evaluation/llm_judge_evaluation.csv'); h=[3,1,2,4,5,5,4,2,3,4,5,3,2,3,4,5,3,4,5,4,2,1,2,2,3,3,1,2,1,3]; print('Human mean:',sum(h)/len(h)); print('LLM mean:',a['overall'].mean()); print('Exact agreement:',sum(x==y for x,y in zip(h,a['overall']))/30); print('Cohen kappa:',cohen_kappa_score(h,a['overall']))"
Human mean: 3.033333333333333
LLM mean: 2.0
Exact agreement: 0.3
Cohen kappa: 0.11016949152542377
PS C:\Users\SOWMYA KARTHIKEYAN\OneDrive\Documents\Desktop\HIVER>Get-Content README.md
# Hiver AI Customer Support Agent

An AI-assisted customer support agent built as a take-home SDE assignment.

The system analyzes historical customer-support conversations, classifies incoming customer issues, retrieves similar historical support cases, and decides whether to answer, clarify, or escalate.

---

## 1. Problem

Customer-support teams handle large volumes of repetitive requests across areas such as:

- Account access
- Billing and payments
- Premium subscriptions
- Playback problems
- Application issues
- Device/platform issues
- Content availability
- Playlists and libraries
- Feature requests
- Account security

The goal of this project is to build a small, explainable support-agent pipeline that uses historical support interactions while avoiding unsupported answers when the available evidence is weak. 

---

## 2. Approach

The overall pipeline is:

```text
Historical Support Dataset
          |
          v
Data Audit & Cleaning
          |
          v
Conversation Reconstruction
          |
          v
Brand Selection
          |
          v
SpotifyCares Support Data
          |
          v
Intent Discovery
          |
          v
Human-Labeled Golden Set
          |
          +--------------------+
          |                    |
          v                    v
   Intent Classifier     Historical Retriever
          |                    |
          +---------+----------+
                    |
                    v
             Decision Engine
                    |
          +---------+---------+
          |         |         |
          v         v         v
       ANSWER   CLARIFY   ESCALATE
                    |
                    v
              FastAPI API
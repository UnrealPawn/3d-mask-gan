import xlrd
from xlutils.copy import copy
excle_dir = './excel.xls'
excle = excle_dir
rb = xlrd.open_workbook(excle_dir, formatting_info=True)
excleCopy = copy(rb)
def saveValue(str,epoch,d_loss,g_loss,d_acc):

     sheet = excleCopy.get_sheet(0)
     if str == 'D':
         sheet.write(epoch, 0, epoch)
         sheet.write(epoch, 1 ,'D training')
         sheet.write(epoch, 2 , d_loss)
         sheet.write(epoch, 3 , g_loss)
         sheet.write(epoch, 4 , d_acc)

     excleCopy.save(excle_dir)
     if str == 'G':
         sheet.write(epoch, 5, 'G training')
         sheet.write(epoch, 6, d_loss)
         sheet.write(epoch, 7, g_loss)
         sheet.write(epoch, 8, d_acc)

     excleCopy.save(excle_dir)

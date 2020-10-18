from sentimentanalysismodel import sentiment_analysis
from termcolor import colored


if __name__ == '__main__':
    print(colored('مرحبا بك :) من فضلك استخدم اللغة العربية', 'yellow'))
    print('----------------------------------------')
    print('\n')

    print(colored('ما أسم حضرتك ؟؟', 'green'))
    name = str(input())
    print('\n')
    print(colored(' اهلا بك {}'.format(name), 'green'))


    while True:
    
        print('\n')
        print(colored('ما أسم الفندق الذي تود تقييمه ؟؟' , 'green'))
        hotel = str(input())

        print('\n')
        print(colored('كم عدد ليالي الأقامه ؟؟', 'green'))
        number_of_night = str(input())

        print('\n')
        print(colored('ما هو تعليقك علي الفندق ؟؟' , 'green'))
        review = str(input())

        print('\n')
        print(colored('من فضلك ادخل تقيم من 1الي 5 للفندق' , 'green'))
        rating = str(input())

        print('\n')
        print(colored(' شكرا لتقيمك فندق {}'.format(hotel) , 'green'))

          
        result = sentiment_analysis(review)   
     
        if result == 'Positive':
            print(colored('كان هذا تعليق اجابي' , 'blue'))
        else:
            print(colored('كان هذا تعليق سلبي' , 'red'))    

        print('\n')
        print(colored('هل تود تقيم فندق اخر نعم/لا', 'green'))
        res = str(input())    

        if res == 'لا':
            break

  
       


# this is Gui for chat boot 
"""
def send():
    
    ChatLog.insert(END, "الالي: " + u'مرحبا بك !!!' + '\n\n')
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12 ))

        res = chatbot_response(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n')

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)


base = Tk()
base.title("مرحبا بك !!!")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)

#Create Chat window
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial",)

ChatLog.config(state=DISABLED)

#Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

#Create Button to send message
SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
                    command= send )

#Create the box to enter message
EntryBox = Text(base, bd=0, bg="white",width="29", height="5", font="Arial")
#EntryBox.bind("<Return>", send)


#Place all components on the screen
scrollbar.place(x=376,y=6, height=386)
ChatLog.place(x=6,y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)

base.mainloop()
"""

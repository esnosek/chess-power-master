# -*- coding: utf-8 -*-

def show_samples():
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    import os
    
    # Parameters for our graph; we'll output images in a 4x4 configuration
    nrows = 4
    ncols = 4
    
    pic_index = 0 # Index for iterating over images
    
    # Set up matplotlib fig, and size it to fit 4x4 pics
    fig = plt.gcf()
    fig.set_size_inches(ncols*4, nrows*4)
    
    pic_index+=8
    
    base_dir = '../data/train_data/empty_or_occupied_50x50'
    
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')
    
    # Directory with our training cat/dog pictures
    train_empty_dir = os.path.join(train_dir, 'EMPTY')
    train_occupied_dir = os.path.join(train_dir, 'OCCUPIED')
    
    train_empty_fnames = os.listdir( train_empty_dir )
    train_occupied_fnames = os.listdir( train_occupied_dir )
    
    next_empty_pix = [os.path.join(train_empty_dir, fname) 
                    for fname in train_empty_fnames[ pic_index-8:pic_index] 
                   ]
    
    next_occupied_pix = [os.path.join(train_occupied_dir, fname) 
                    for fname in train_occupied_fnames[ pic_index-8:pic_index]
                   ]
    
    for i, img_path in enumerate(next_empty_pix + next_occupied_pix):
      # Set up subplot; subplot indices start at 1
      sp = plt.subplot(nrows, ncols, i + 1)
      sp.axis('Off') # Don't show axes (or gridlines)
    
      img = mpimg.imread(img_path)
      plt.imshow(img)
    
    plt.show()

def predict():
    
    import os
    from tensorflow.keras.preprocessing import image
    import numpy as np
    
    base_dir = '../data/train_data/empty_or_occupied_50x50'
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')
    
    for n, file in enumerate(os.scandir(os.path.join(validation_dir, 'OCCUPIED'))):
        # predicting images
        filename, file_extension = os.path.splitext(file.path)
        
        img=image.load_img(file.path, target_size=(50, 50))
        x=image.img_to_array(img)
        x=np.expand_dims(x, axis=0)
        images = np.vstack([x])
          
        classes = model.predict(images, batch_size=10)
          
        if classes[0]>0:
            print(file.name + " is occupied")
        else:
            print(file.name + " is empty")

def plot_history():
    #-----------------------------------------------------------
    # Retrieve a list of list results on training and test data
    # sets for each training epoch
    #-----------------------------------------------------------
    acc      = history.history[     'acc' ]
    val_acc  = history.history[ 'val_acc' ]
    loss     = history.history[    'loss' ]
    val_loss = history.history['val_loss' ]
    
    epochs   = range(len(acc)) # Get number of epochs
    
    #------------------------------------------------
    # Plot training and validation accuracy per epoch
    #------------------------------------------------
    plt.plot  ( epochs,     acc )
    plt.plot  ( epochs, val_acc )
    plt.title ('Training and validation accuracy')
    plt.figure()
    
    #------------------------------------------------
    # Plot training and validation loss per epoch
    #------------------------------------------------
    plt.plot  ( epochs,     loss )
    plt.plot  ( epochs, val_loss )
    plt.title ('Training and validation loss'   )

predict()    
print(model)
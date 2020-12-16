import React,{useRef} from 'react';
import {SketchField, Tools} from 'react-sketch';
import { Button } from 'react-bootstrap';
import { saveAs } from 'file-saver';

const styles={
    draw: {
        margin :'140px'
    }
}

const Draw = () => {

    const sketch = useRef()

    const handleSubmit = () => {
        const canvas =sketch.current.toDataURL()
        saveAs(canvas, 'pixmap.jpg')
        sendData(canvas)
    }

    const handelReset =() => {
        sketch.current.clear() 
        sketch.current._backgroundColor('grey')
    }

    const sendData = (c) => {
        console.log(c)
    }
    const getImageResult = (id) =>{
        
    }
    return (
        <React.Fragment>
            <h1>My Canvas</h1>
            <SketchField ref={sketch}
                         width='750px' 
                         height='750px' 
                         style={styles.draw}
                         tool={Tools.Pencil} 
                         backgroundColor='grey'
                         lineColor='white'
                         imageFormat='jpg'
                         lineWidth={40}
            />
            <div className="mt-3">
                <Button onClick={handleSubmit} variant ='primary'>Save</Button>
                <Button onClick={handelReset} variant ='secondary'>Reset</Button>
            </div>
        </React.Fragment>

     );
}


export default Draw;
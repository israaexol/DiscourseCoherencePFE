import { useState } from 'react';
import * as  React from 'react';
import axios from 'axios'
import { Form, Row, Col } from "react-bootstrap";
import Button from '@mui/material/Button';
import ButtonGroup from '@mui/material/ButtonGroup';
import ArrowDropDownIcon from '@mui/icons-material/ArrowDropDown';
import ClickAwayListener from '@mui/material/ClickAwayListener';
import Grow from '@mui/material/Grow';
import Paper from '@mui/material/Paper';
import Popper from '@mui/material/Popper';
import MenuItem from '@mui/material/MenuItem';
import MenuList from '@mui/material/MenuList';
import './Accueil.css'
import Sidebar from '../Sidebar/Sidebar'
import Result from '../Result/Result'
import Box from '@mui/material/Box';
import CircularProgress from '@mui/material/CircularProgress';

const Accueil = () => {

  const [text, setText] = useState("");
  const options = ['Parenté sémantique entre les phrases', 'Parenté sémantique entre les paragraphes', 'Parenté sémantique entre les phrases et les paragraphes', 'Richesse lexicale', 'Richesse lexicale et parenté sémantique'];
  const [open, setOpen] = React.useState(false);
  const anchorRef = React.useRef(null);
  const [selectedIndex, setSelectedIndex] = React.useState(0);
  const [isLoading, setLoading] = useState(null)

  const handleSubmit = (event) => {
    setLoading(true)
    event.preventDefault();
    // alert(` you entered : ${text}, ${selectedIndex}`);
    const params = { text, selectedIndex };
    var divelement = document.getElementById('evalSection')
    axios
      .post('http://localhost:8080/evaluate/', params)
      .then((res) => {
        const data = res.data.data
        const msg = `${data.score}`
        divelement.hidden = false
        setScore(msg)
        setLoading(false)
      })
      .catch((error) => {
        // alert(`Error: ${error.message}`)
        divelement.hidden = false
        setScore(error.message)
        setLoading(false)
      })

  }

  const handleClick = () => {
  };

  const handleMenuItemClick = (event, index) => {
    setSelectedIndex(index);
    setOpen(false);
  };

  const handleToggle = () => {
    setOpen((prevOpen) => !prevOpen);
  };

  const handleClose = (event) => {
    if (anchorRef.current && anchorRef.current.contains(event.target)) {
      return;
    }

    setOpen(false);
  };
  const [scoreResult, setScore] = useState();
  
  function RenderResult({isLoading}) {
    if(isLoading === null) {
      return <Result hidden={true}/>
    }
    else if(isLoading === true) {
      return (
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', marginTop: '4%' }}>
        <CircularProgress/>
        <Result hidden={true}/>
        </Box>
          )
    }
    else {
      return <Result hidden={false} scoreResult={scoreResult} />
    }
      
  }

  return (
    <>
      <Sidebar selectedIndex={selectedIndex}/>
      <div className='form'>
        <Form onSubmit={handleSubmit}>
          <div className='input_text'>
            <textarea
              id='CheckIt'
              className='_textarea'
              required
              type='text'
              placeholder="Insérez votre texte" 
              value={text}
              onChange={(e) => setText(e.target.value)
              }
            />
          </div>
          <br />
          <Button type="button" id='import_btn'>Importer un fichier</Button>

          <div className='eval_anal'>
            <div id='analyser_btn'>
              <ButtonGroup variant="contained" ref={anchorRef} aria-label="split button">
                <Button onClick={handleClick} resize="none" >{options[selectedIndex]}</Button>
                <Button
                  size="small"
                  aria-controls={open ? 'split-button-menu' : undefined}
                  aria-expanded={open ? 'true' : undefined}
                  aria-label="select merge strategy"
                  aria-haspopup="menu"
                  onClick={handleToggle}
                >
                  <ArrowDropDownIcon />
                </Button>
              </ButtonGroup>
              <Popper
                open={open}
                anchorEl={anchorRef.current}
                role={undefined}
                transition
                disablePortal
              >
                {({ TransitionProps, placement }) => (
                  <Grow
                    {...TransitionProps}
                    style={{
                      transformOrigin:
                        placement === 'bottom' ? 'center top' : 'center bottom',
                    }}
                  >
                    <Paper>
                      <ClickAwayListener onClickAway={handleClose}>
                        <MenuList id="split-button-menu" autoFocusItem>
                          {options.map((option, index) => (
                            <MenuItem
                              key={option}
                              selected={index === selectedIndex}
                              onClick={(event) => handleMenuItemClick(event, index)}
                            >
                              {option}
                            </MenuItem>
                          ))}
                        </MenuList>
                      </ClickAwayListener>
                    </Paper>
                  </Grow>
                )}
              </Popper>
            </div>
            <Button type="submit" id='eval_btn'>Évaluer</Button>
          </div>
          <RenderResult isLoading={isLoading}/>
        </Form>
      </div>
    </>

  )
}

export default Accueil
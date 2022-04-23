import { useState } from 'react';
import * as  React from 'react';
import axios from 'axios'
import { Form, Row, Col, Stack } from "react-bootstrap";
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
import ThreeSixtyIcon from '@mui/icons-material/ThreeSixty';

import { BarChart, Bar, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';


const Accueil = () => {

  const [text, setText] = useState("");
  const options = ['Parenté sémantique entre les phrases', 'Parenté sémantique entre les paragraphes', 'Parenté sémantique entre les phrases et les paragraphes', 'Richesse lexicale', 'Richesse lexicale et parenté sémantique'];
  const [open, setOpen] = React.useState(false);
  const anchorRef = React.useRef(null);
  const hiddenFileInput = React.useRef(null);
  const [selectedIndex, setSelectedIndex] = React.useState(0);
  const [isLoading, setLoading] = useState(null)
  const [isEmpty, setEmpty] = useState(true)
  const [state, setState] = useState(null)
  const [data, setData] = useState(null)
  const [scoreResult, setScore] = useState(null);


  const handleSubmit = (event) => {
    setLoading(true)
    event.preventDefault();
    const params = { text, selectedIndex };
    var divelement = document.getElementById('evalSection')
    if (data == null) {
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
          divelement.hidden = false
          setScore(error.message)
          setLoading(false)
        })
    } else {
      axios
        .post('http://localhost:8080/uploadfile?niveau='+selectedIndex, data)
        .then((res) => {
          const data = res.data.data
          const score = data
          let index = 0
          while(index < score.length) {
            score[index]++;
            index++;
          }
          // alert(score)
          // var myArray = JSON.Parse(score);
          let chart_result = []
          for (var i = 0; i < score.length; i++) {
            let label = "doc" + i
            let obj = {
              label: label,
              score: score[i]
            }
            chart_result.push(obj)
          }

          setState(chart_result)
          setEmpty(false)
          setLoading(false)
        })
        .catch((error) => {
          alert(`Error: ${error.message}`)
          divelement.hidden = false
          setScore(error.message)
          setLoading(false)
        })
    }

  }
  const handleImport = event => {
    var textarea = document.getElementById('CheckIt');
    textarea.required = false;
    textarea.disabled = true;
    hiddenFileInput.current.click();

  };
  const handleChange = event => {
    const fileUploaded = event.target.files[0];
    if (fileUploaded) {
      let dataFile = new FormData();
      dataFile.append('file', fileUploaded);
      setData(dataFile)
    }
  };


  const handleRefresh = () => {
    window.location.reload();
  };

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

  function RenderResult({ isLoading }) {
    if (isLoading === null) {
      return <Result hidden={true} />
    }
    else if (isLoading === true) {
      return (
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', marginTop: '4%' }}>
          <CircularProgress />
          <Result hidden={true} />
        </Box>
      )
    }
    else {
      return <Result hidden={false} scoreResult={scoreResult} />
    }

  }

  function RenderChart({ isEmpty }) {
    if (isEmpty === true) {
      return <></>
    }
    else {
      return (
        <BarChart
          width={500}
          height={300}
          data={state}
          margin={{
            top: 5,
            right: 30,
            left: 20,
            bottom: 5,
          }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="label" />
          <YAxis tickCount={4} />
          <Tooltip />
          <Legend />
          <Bar dataKey="score" fill="#8884d8" />
        </BarChart>
      )
    }
  }

  return (
    <>
      <Sidebar selectedIndex={selectedIndex} />
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
          <div className="file-inputs">
            <Button type="button" id='import_btn' onClick={handleImport}>Importer un fichier</Button>
            <input type="file" ref={hiddenFileInput} onChange={handleChange} style={{ display: 'none' }} />
            <Button variant="outlined" startIcon={<ThreeSixtyIcon />} onClick={handleRefresh}>
              Rafraîchir
            </Button>
          </div>

          {/* <Button type="button" id='import_btn' onClick={handleImport}>Importer un fichier</Button> */}

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
          <RenderResult isLoading={isLoading} />
          <RenderChart isEmpty={isEmpty} />
        </Form>
      </div>
    </>

  )
}

export default Accueil
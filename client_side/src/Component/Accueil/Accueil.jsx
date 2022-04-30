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
import Typography from '@mui/material/Typography';
import LooksOneIcon from '@mui/icons-material/LooksOne';
import LooksTwoIcon from '@mui/icons-material/LooksTwo';

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
  const [chartLength, setChartLength] = useState(0);
  const [array_cell, setArray] = useState(null);
  const [fileName, setFileName] = useState(null)

  function createData(text_id, text, original_score, predicted_score) {
    return { text_id, text, original_score, predicted_score };
  }

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
          let msg = data.score
          msg++
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
        .post('http://localhost:8080/uploadfile?niveau=' + selectedIndex, data)
        .then((res) => {
          const data = res.data.data
          const score = data.scores
          let table_details = []
          setChartLength(score.length)
          let index = 0
          while (index < score.length) {
            score[index]++;
            index++;
          }
          // alert(score)
          // var myArray = JSON.Parse(score);
          let chart_result = [
            {
              label: 'Low',
              score: 0
            },
            {
              label: 'Medium',
              score: 0
            },
            {
              label: 'High',
              score: 0
            },
          ]
          let count_low = 0
          let count_med = 0
          let count_high = 0
          for (var i = 0; i < score.length; i++) {
            if (score[i] == 1) {
              count_low++
            } else if (score[i] == 2) {
              count_med++
            } else {
              count_high++
            }
            let cell = createData(data.text_ids[i], data.texts[i], data.original_scores[i], score[i])
            table_details.push(cell)

          }
          chart_result[0].score = count_low
          chart_result[1].score = count_med
          chart_result[2].score = count_high
          setArray(table_details)
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
      setFileName(fileUploaded.name)
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
      if (state === null) {
        return <Result hidden={false} scoreResult={scoreResult} isEmpty={true} chartData={null} chart={false} table={null} />
      }
      else {
        return <Result hidden={false} scoreResult={null} isEmpty={false} chartData={state} chart={true} chartLength={chartLength} table={array_cell} />
      }

    }

  }

  function Item(props) {
    const { sx, ...other } = props;
    return (
      <Box
        sx={{
          p: 1,
          bgcolor: 'transparent',
          color: (theme) => (theme.palette.mode === 'dark' ? 'grey.300' : 'grey.800'),
          fontSize: '0.875rem',
          ...sx,
        }}
        {...other}
      />
    );
  }

  return (
    <>
      <Sidebar selectedIndex={selectedIndex} />
      <div id="firstSection">
        <div style={{ marginTop: '1%' }} >
          <Box
            sx={{
              display: 'flex',
              flexDirection: 'row',
              p: 1,
              justifyContent: 'space-between',
              position: 'relative',
              width: '75%',
              height: '60px',
              margin: '1px 133px'
            }}
          >
            <Item sx={{ backgroundColor: 'none', height: '50px', width: '100%' }}>
              <Typography variant="h5" sx={{ fontFamily: 'Poppins', fontWeight: 500, color: '#5885FB' }}><LooksOneIcon sx={{ margin: '0 18px', height: '6%', width: '6%', color: "#ffab00" }} />Insertion des données</Typography>
            </Item>
            <Item sx={{ backgroundColor: 'none', marginRight: '10%' }}>
              <Button variant="outlined" startIcon={<ThreeSixtyIcon />} onClick={handleRefresh}>
                Rafraîchir
              </Button>
            </Item>
          </Box>
        </div>
      </div>
      <div className='form'>
        <Form onSubmit={handleSubmit}>
          <div className='input_text'>
            <textarea
              id='CheckIt'
              className='_textarea'
              required
              type='text'
              placeholder="Insérez un texte"
              value={text}
              onChange={(e) => setText(e.target.value)
              }
            />
          </div>
          <br />
          <Box
            sx={{
              display: 'flex',
              flexDirection: 'row-reverse',
              p: 1,
              justifyContent: 'flex-start',
              position: 'absolute',
              width: '63%',
              height: '80px',
              marginTop: '-1%',
              textAlign: 'center'
            }}
          >
            <Item >
              <div className="file-inputs">
                <Button type="button" id='import_btn' onClick={handleImport}> {fileName ?? "Importer un fichier"} </Button>
                <input type="file" ref={hiddenFileInput} onChange={handleChange} style={{ display: 'none' }} />
              </div>
            </Item>
            <Item><Typography sx={{ fontFamily: 'Poppins', fontSize: '16px', padding: '50% 0px' }}>Ou</Typography></Item>
          </Box>

          {/* <Button type="button" id='import_btn' onClick={handleImport}>Importer un fichier</Button> */}

          <div id="secondSection">
            <div>
              <Box
                sx={{
                  display: 'flex',
                  flexDirection: 'row',
                  justifyContent: 'space-between',
                  position: 'relative',
                  width: '75%',
                  height: '60px',
                  margin: '10px 133px'
                }}
              >
                <Item sx={{ backgroundColor: 'none', height: '50px', width: '100%' }}>
                  <Typography variant="h5" sx={{ fontFamily: 'Poppins', fontWeight: 500, color: '#5885FB' }}><LooksTwoIcon sx={{ margin: '0 18px', height: '5%', width: '5%', color: "#ffab00" }} />Sélection du modèle</Typography>
                </Item>
              </Box>
            </div>
          </div>

          <div className='eval_anal'>

            <div id='analyser_btn'>
              <ButtonGroup variant="contained" ref={anchorRef} aria-label="split button" sx={{ zIndex: 1 }}>
                <Button onClick={handleClick} sx={{ width: '400px' }} >{options[selectedIndex]}</Button>
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
        </Form>
      </div>
    </>

  )
}

export default Accueil
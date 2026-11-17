// Optional Chromium regression checks; kept outside the CRAN package.
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const { chromium } = require('playwright');
const url = process.env.DDESONN_APP_URL || 'http://127.0.0.1:3838/';
const output = process.env.DDESONN_BROWSER_OUTPUT || '/tmp/ddesonn-browser-tests';
const allMethods = ['adagrad', 'adam', 'lamb', 'sgd', 'sgd_momentum', 'nag', 'rmsprop', 'ftrl', 'lookahead', 'adadelta'];
const methods = process.env.DDESONN_OPTIMIZERS ? process.env.DDESONN_OPTIMIZERS.split(',') : allMethods;
assert.ok(methods.length && methods.every(m => allMethods.includes(m)), 'Unknown optimizer');
async function enabled(page, id) {
  await page.waitForFunction(id => { const e = document.getElementById(id); return e && !e.disabled; }, id);
}
async function select(page, id, value) {
  await page.evaluate(({id,value}) => {
    const e = document.getElementById(id);
    if (e.selectize) e.selectize.setValue(value);
    else { e.value = value; e.dispatchEvent(new Event('change', {bubbles:true})); }
  }, {id,value});
}
(async () => {
  await fs.mkdir(output, {recursive:true});
  const browser = await chromium.launch({headless:true,
    executablePath:process.env.CHROMIUM_PATH || '/usr/bin/chromium', args:['--no-sandbox']});
  const errors = [];
  const optionalBrowserRequests = [];
  const context = await browser.newContext({viewport:{width:1366,height:1000},acceptDownloads:true});
  const page = await context.newPage();
  page.on('pageerror', e => errors.push(e.message));
  page.on('console', m => {
    if (m.type() !== 'error') return;
    const entry = {message:m.text(),location:m.location()};
    // Chromium probes this optional icon even though the app does not declare one.
    if (new URL(entry.location.url || url).pathname === '/favicon.ico' && /404/.test(entry.message))
      optionalBrowserRequests.push(entry);
    else errors.push(entry);
  });
  page.on('response', r => { if(r.status() >= 400) console.log('HTTP failure', r.status(), r.url()); });
  page.setDefaultTimeout(30000);
  try {
    await page.goto(url);
    await page.waitForFunction(() => window.Shiny?.shinyapp?.$socket?.readyState === 1);
    await page.waitForFunction(() => document.querySelector('#wb_flow_guide')?.textContent.includes('Train a new model:'));
    await page.locator('input[name="wb_task_flow"][value="saved"]').check();
    await page.getByText('R example for prediction', {exact:true}).click();
    await page.locator('#wb_model_code').waitFor({state:'visible'});
    await page.waitForFunction(() => document.querySelector('#wb_model_code')?.textContent.includes('readRDS('));
    assert.ok((await page.locator('#wb_model_code').textContent()).includes('library(DDESONN)'));
    await page.locator('input[name="wb_task_flow"][value="train"]').check();
    await page.waitForFunction(() => document.querySelector('#wb_flow_guide')?.textContent.includes('Train a new model:'));
    await page.click('#wb_next_features');
    await enabled(page, 'wb_prepare');
    await page.click('#wb_prepare');
    await page.waitForFunction(() => document.querySelector('#wb_prepared')?.textContent.includes('90 rows'));
    await page.locator('a[data-value="train"]').click();
    await page.fill('#wb_hidden', '2');
    await page.fill('#wb_epochs', '1');
    await page.locator('#wb_epochs').press('Tab');
    const checkpoints = await page.locator('select').evaluateAll(es => es.filter(e=>e.id.includes('checkpoint')).map(e=>({id:e.id,options:[...e.options].map(o=>o.value)})));
    console.log('checkpoint controls', JSON.stringify(checkpoints));
    assert.equal(await page.locator('#wb_average_gradients').isChecked(), false);
    await page.locator('details:has(#wb_training_code) > summary').click();
    for (const [index, method] of methods.entries()) {
      const average = index % 2 === 1;
      if (await page.locator('#wb_average_gradients').isChecked() !== average)
        await page.locator('label.wb-switch:has(#wb_average_gradients)').click();
      await page.waitForFunction(average => document.querySelector('#wb_training_code')?.textContent.includes(
        `average_gradients = ${average ? 'TRUE' : 'FALSE'}`), average);
      await select(page, 'wb_optimizer', method);
      const checkpoint = checkpoints.find(e=>e.options.includes('best_validation'));
      if (checkpoint) await select(page, checkpoint.id, index % 2 ? 'best_validation' : 'final');
      await enabled(page, 'wb_start');
      await page.click('#wb_start');
      await page.waitForFunction(() => document.querySelector('#wb_cancel')?.disabled === false);
      await page.waitForFunction(() => {
        const s=document.querySelector('#wb_status')?.textContent || '';
        return /Completed|Failed/.test(s) && document.querySelector('#wb_cancel')?.disabled;
      }, null, {timeout:90000});
      const status=await page.locator('#wb_status').textContent();
      assert.match(status,/Completed/,status);
      await page.waitForFunction(method => document.querySelector('#wb_model_info')?.textContent.includes(method), method);
      const log=await page.locator('#wb_training_log').textContent();
      assert.match(log,/Epoch 1 .*Loss:/);
      assert.doesNotMatch(log,/\[OPT=|Grad norm|\[LEARN\]|FUSE-DBG/);
      await page.waitForFunction(average => document.querySelector('#wb_checkpoint_summary')?.textContent.includes(
        average ? 'averaged gradients' : 'summed gradients'), average);
      console.log(`${method}: browser training completed, clean progress log; average_gradients=${average}`);
    }
    const retainedModel = await page.locator('#wb_model_info').textContent();
    await page.fill('#wb_epochs', '5000');
    await page.locator('#wb_epochs').press('Tab');
    await page.click('#wb_start');
    await enabled(page, 'wb_cancel');
    await page.click('#wb_cancel');
    await page.waitForFunction(()=>/cancelled/i.test(document.querySelector('#wb_status')?.textContent || ''));
    await enabled(page, 'wb_start');
    assert.equal(await page.locator('#wb_model_info').textContent(), retainedModel);
    console.log('cancelled background training; previous completed model remains available');
    await page.screenshot({path:path.join(output,'desktop-training.png'),fullPage:true});
    await page.locator('a[data-value="results"]').click();
    await page.locator('#wb_metrics tbody tr').first().waitFor({state:'visible'});
    await page.locator('a[data-value="train"]').click();
    await page.locator('#wb_save_selected').waitFor({state:'visible'});
    const downloadEvent=page.waitForEvent('download');
    await page.click('#wb_save_selected');
    const download=await downloadEvent;
    const bundle=path.join(output,'browser-model.rds');
    await download.saveAs(bundle);
    assert.ok((await fs.stat(bundle)).size>0);
    await page.locator('input[name="wb_task_flow"][value="saved"]').check();
    await page.locator('#wb_load').setInputFiles(bundle);
    await page.waitForFunction(()=>document.querySelector('#wb_active_model')?.textContent.includes('browser-model.rds'));
    await page.waitForFunction(()=>document.querySelector('#wb_saved_evaluation')?.textContent.includes('Training'));
    const csv = path.join(output, 'prediction.csv');
    await fs.writeFile(csv, 'Sepal.Length,Sepal.Width,Petal.Length,Petal.Width\n5.1,3.5,1.4,0.2\n6.7,3.0,5.2,2.3\n');
    await page.locator('input[name="wb_csv_purpose"][value="predict"]').check();
    await page.locator('#wb_new_data').setInputFiles(csv);
    await enabled(page, 'wb_predict');
    await page.click('#wb_predict');
    await page.waitForFunction(()=>document.querySelector('#wb_new_preview')?.querySelectorAll('tbody tr').length === 2);
    const predictions = await page.locator('#wb_new_preview').textContent();
    const rendered = page.evaluate(() => new Promise(resolve => {
      const handler = event => { if(event.name === 'wb_new_preview') { $(document).off('shiny:value', handler); resolve(); } };
      $(document).on('shiny:value', handler);
    }));
    await page.click('#wb_predict');
    await rendered;
    assert.equal(await page.locator('#wb_new_preview').textContent(), predictions);
    assert.ok(await page.locator('#wb_new_chart img').getAttribute('src'));
    console.log('uploaded CSV produces two repeatable predictions and a chart');
    await page.locator('a[data-value="continue"]').click();
    await page.click('#wb_epochs_clear');
    await page.waitForFunction(()=>document.querySelector('#wb_continue').disabled);
    await page.click('#wb_epochs_10');
    await enabled(page,'wb_continue');
    assert.equal(await page.locator('#wb_more_epochs').inputValue(),'10');
    await page.fill('#wb_more_epochs','1');
    await page.locator('#wb_more_epochs').press('Tab');
    await page.click('#wb_continue');
    await page.waitForFunction(()=>document.querySelector('#wb_saved_fit_status')?.textContent.startsWith('Completed'), null, {timeout:90000});
    await enabled(page,'wb_saved_use_fit');
    assert.match(await page.locator('#wb_active_model').textContent(), /browser-model.rds/);
    console.log('additional training completed as a separate copy');
    await page.screenshot({path:path.join(output,'desktop-saved-model.png'),fullPage:true});
    await page.setViewportSize({width:390,height:844});
    // Shiny sends responsive plot sizes back to R after a viewport resize.
    await page.waitForFunction(()=>document.documentElement.scrollWidth <= innerWidth + 1);
    await page.screenshot({path:path.join(output,'mobile-saved-model.png'),fullPage:true});
    const sizes=await page.evaluate(()=>({width:innerWidth,scroll:document.documentElement.scrollWidth}));
    assert.ok(sizes.scroll<=sizes.width+1,JSON.stringify(sizes));
    // Keyboard reaches the additional-training input and retains its label.
    await page.locator('#wb_more_epochs').focus();
    await page.keyboard.press('Tab');
    assert.notEqual(await page.evaluate(()=>document.activeElement.tagName),'BODY');
    assert.deepEqual(errors,[]);
    console.log('model download/reload, epoch controls, desktop/mobile overflow and keyboard checks passed');
    await fs.writeFile(path.join(output,'result.json'),JSON.stringify({methods,errors,optionalBrowserRequests,mobile:sizes,passed:true},null,2));
  } catch (e) {
    await page.screenshot({path:path.join(output,'failure.png'),fullPage:true});
    console.error(e);
    console.error('overflow elements', await page.evaluate(()=>[...document.querySelectorAll('body *')].filter(e=>e.getBoundingClientRect().right>innerWidth+1&&e.getBoundingClientRect().width>0).map(e=>({tag:e.tagName,id:e.id,cls:String(e.className),width:e.getBoundingClientRect().width,right:e.getBoundingClientRect().right,text:e.textContent?.slice(0,45)})).slice(0,30)));
    process.exitCode=1;
  } finally { await browser.close(); }
})();
